# Experience Compiler Lab — Original Specification (verbatim)

> Transcribed from the kickoff message (Telegram, 2026-08-29). The initial
> block duplicates fragments of §3/§7 — retained verbatim. This file is the
> ground-truth reference; every downstream document cites it by section.

---

┌─────┴─────┐
                       ▼           ▼
                    promote      reject
                       │           │
                       └─────┬─────┘
                             ▼
                         ledger

The execution agent should not receive the full knowledge base.
It receives only:
system instructions
active skills
task
tool interfaces

The Knowledge Miner and Skill Proposer receive historical evidence.
This intentionally keeps learning-time knowledge separate from execution-time knowledge.

## 4. Repository layout

```
experience-compiler-lab/
├── README.md
├── pyproject.toml
├── src/
│   ├── agent/
│   │   ├── executor.py
│   │   └── context.py
│   ├── world/
│   │   ├── api.py
│   │   ├── state.py
│   │   └── fixtures/
│   ├── traces/
│   │   ├── schema.py
│   │   └── store.py
│   ├── knowledge/
│   │   ├── miner.py
│   │   ├── schema.py
│   │   └── store.py
│   ├── skills/
│   │   ├── proposer.py
│   │   ├── loader.py
│   │   └── patch.py
│   ├── evals/
│   │   ├── runner.py
│   │   ├── graders.py
│   │   └── metrics.py
│   └── experiments/
│       └── evolution_loop.py
│
├── experience/
│   └── runs/
├── knowledge/
│   ├── patterns/
│   └── index.yaml
├── skills/
│   └── onboarding/
│       ├── SKILL.md
│       └── PURPOSE.yaml
├── datasets/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── results/
└── notebooks/
```

## 5. Trace schema

Avoid storing only transcripts.
Represent trajectories explicitly:

```python
class Trace:
    run_id: str
    task_id: str
    model: str
    skill_version: str
    messages: list[Message]
    actions: list[Action]
    final_answer: str | None
    outcome: Outcome
    metrics: Metrics
```

An action should include:

```python
class Action:
    index: int
    tool: str
    arguments: dict
    result: dict
    timestamp: datetime
    world_state_before: str
    world_state_after: str
```

The final outcome can initially be deterministic:

```python
class Outcome:
    success: bool
    errors: list[str]
    violated_constraints: list[str]
```

This is important because the system should reason over observable actions,
not just natural-language reasoning traces.

## 6. Knowledge representation

Do not copy WikiSkill's unconstrained Markdown directly.
Use structured records with an optional human-readable Markdown rendering.

```yaml
id: inventory-check-before-assignment
claim:
  type: procedure
  text: >
    Check available inventory before assigning hardware.
scope:
  workflows:
    - onboarding
evidence:
  supporting_runs:
    - run_014
    - run_031
  counterexamples:
    - run_022
statistics:
  support: 8
  failures: 2
confidence: 0.82
status: active
first_seen: 2026-08-29
last_updated: 2026-08-29
```

Later you can extend this with:
affected_models
affected_tools
environment_version
supersedes
contradicts
skill_dependencies

## 7. Evidence Miner

The Evidence Miner receives batches of trajectories.
Its prompt should ask it to find:
- repeated failure modes
- repeated successful strategies
- incorrect assumptions
- missing checks
- bad tool sequences
- unnecessary calls
- termination failures
- recovery strategies

It must output structured candidate evidence.
The system then merges candidates deterministically where possible.

Example:
Run 12:
assign_laptop("Alice", "MBP-42")
→ inventory error
Run 18:
assign_laptop(...)
→ inventory error
Run 20:
get_inventory()
assign_laptop(...)
→ success

Miner output:
hypothesis:
  check inventory before assignment
support:
  [run12, run18, run20]

The LLM generates the hypothesis.
The evidence links remain deterministic.

## 8. Skill format

Keep skills simple.

```markdown
# Employee Onboarding
## Objective
Complete onboarding while respecting workflow dependencies.
## Procedure
1. Resolve employee identity.
2. Retrieve onboarding requirements.
3. Check hardware inventory.
4. Assign available hardware.
5. Request procurement if inventory is unavailable.
6. Provision required access.
7. Verify all mandatory tasks.
8. Mark onboarding complete only when all blockers are cleared.
## Failure handling
If a required dependency cannot be fulfilled:
- record the blocker;
- escalate to the responsible team;
- leave the workflow incomplete.
```

And separately:

```yaml
skill: onboarding
version: 4
derived_from:
  - inventory-check-before-assignment
  - incomplete-access-before-close
proposed_by:
  model: claude-x
evaluation:
  previous_score: 0.71
  candidate_score: 0.79
status: accepted
```

PURPOSE.yaml is valuable because it creates explicit provenance.

## 9. Skill Proposer

Input:
- active skill
- structured knowledge
- recent accepted/rejected proposals
- relevant traces

Output:

```diff
@@ Procedure
- 3. Assign required hardware.
+ 3. Check hardware inventory.
+ 4. Assign hardware only if inventory confirms availability.
+ 5. Otherwise create a procurement blocker and escalate.
```

Prefer patches over full skill regeneration.
This lets you measure:
- size of intervention
- location changed
- reason for intervention
- regression impact

---

# New project to develop

Build the PoC as an "Agent Experience Compiler": a small harness that turns agent
execution traces into persistent evidence, extracts reusable patterns, proposes
skill patches, validates them independently, and promotes only improvements.
Keep the first version deliberately narrow: one agent framework, one tool-heavy
benchmark, Markdown/YAML artifacts, deterministic evaluation, Git-backed history.
The research question is whether structured experience → validated skill
synthesis improves agent performance more reliably than trace-only or
memory-only approaches.

**Project name:** experience-compiler-lab

**Core thesis:**
Agent executions should not directly mutate prompts.

```
execution traces
      ↓
evidence extraction
      ↓
persistent knowledge
      ↓
candidate procedural artifact
      ↓
independent evaluation
      ↓
promotion / rejection
```

The system should preserve the distinction between three stores:

- `experience/` — immutable observations of what happened
- `knowledge/` — accumulated interpretations of experience
- `skills/` — executable procedural knowledge currently deployed

The key invariant is:
**Experience and evidence are append-only. Skills are mutable and reversible.**
A rejected skill patch therefore still produces useful research evidence.

## 1. Goals

The PoC should answer four questions.

First, can repeated agent failures and successes be compiled automatically into
useful procedural knowledge?

Second, does persistent structured knowledge outperform simply giving the
skill-generator recent execution traces?

Third, can candidate skill changes be validated without contaminating
evaluation with the knowledge used to generate those changes?

Fourth, are evolved skills portable across models, or do they encode
model-specific compensatory behavior?

Do not initially attempt autonomous production agents, general-purpose memory,
semantic skill retrieval, distributed execution, or online continuous learning.

## 2. Target experiment

Choose one environment where trajectories matter and failures are interpretable.
A good first option is a synthetic enterprise-tool environment:

```
World API
├── documents
├── tickets
├── inventory
├── users
└── workflows
```

Tasks might include:
- "Onboard employee Alice."
- "Resolve laptop shortage while completing onboarding."
- "Update the ticket after checking inventory."
- "Find the correct policy before approving access."
- "Escalate when required information is missing."

This is preferable to a pure QA benchmark because skills can describe actual
procedures such as:

When onboarding a user:
1. Resolve employee identity first.
2. Check laptop inventory before creating procurement requests.
3. Never mark onboarding complete while required access tasks remain blocked.
4. Escalate inventory shortages rather than inventing equipment assignments.

That makes evolved knowledge inspectable.

## 3. Architecture

```
                    ┌─────────────────────┐
                    │   Task Dataset      │
                    └─────────┬───────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Execution Agent │
                     └────────┬────────┘
                              │
                              ▼
                    immutable trajectories
                              │
                              ▼
                    ┌─────────────────┐
                    │ Evidence Miner  │
                    └────────┬────────┘
                              │
                              ▼
                       Knowledge Base
                              │
                              ▼
                    ┌─────────────────┐
                    │ Skill Proposer  │
                    └────────┬────────┘
                              │
                              ▼
                       candidate patch
                              │
                              ▼
                    ┌─────────────────┐
                    │ Eval Harness    │
                    └───────┬─────────┘
                        pass │ fail
```

The execution agent should not receive the full knowledge base.
It receives only: system instructions, active skills, task, tool interfaces.
The Knowledge Miner and Skill Proposer receive historical evidence.
This intentionally keeps learning-time knowledge separate from execution-time
knowledge.

## 4. Repository layout

(see tree at top of this file)

## 5. Trace schema

(see §5 at top of this file)

## 6. Knowledge representation

(see §6 at top of this file)

## 7. Evidence Miner

(see §7 at top of this file)

## 8. Skill format

(see §8 at top of this file)

## 9. Skill Proposer

(see §9 at top of this file)

## 10. Validation harness

This is the critical research component.
**Never validate using the training tasks that generated the proposed skill.**
Split tasks into:
- development
- validation
- held-out test

A candidate is evaluated against the same fixed validation set as the current
skill.

The simplest promotion rule:

```python
accept = (
    candidate.success_rate > baseline.success_rate
    and candidate.regressions <= allowed_regressions
)
```

Later use a score vector:
- task success
- constraint violations
- tool calls
- tokens
- latency
- recovery rate
- trajectory length

Do not collapse everything into a single score too early.
A candidate that improves success from 80 → 84% but doubles tool calls may
represent a real architectural regression.

## 11. Evolution loop

```python
skill = load_current_skill()
for iteration in range(N):
    traces = run_tasks(tasks=train_tasks, skill=skill)
    knowledge.update(mine_evidence(traces))
    patch = propose_patch(
        skill=skill,
        knowledge=knowledge,
        proposal_history=history,
    )
    candidate = apply_patch(skill, patch)
    result = evaluate(
        candidate=candidate,
        baseline=skill,
        tasks=validation_tasks,
    )
    record_proposal(patch, result, evidence=knowledge.references)
    if promotion_policy(result):
        skill = candidate
```

The rejected proposal remains in the history.
That is essential. Otherwise iteration 8 may simply repeat the mistake from
iteration 3.

## 12. Experimental baselines

This is where the project becomes publishable-quality applied research rather
than a demo. Compare four configurations:

| Configuration        | Persistent traces | Structured knowledge | Evolved skills |
|----------------------|-------------------|----------------------|----------------|
| Baseline             | no                | no                   | no             |
| Trace2Skill          | yes               | no                   | yes            |
| Memory Agent         | yes               | yes                  | no             |
| Experience Compiler  | yes               | yes                  | yes            |

You want to test:

- **H1:** structured persistent knowledge produces better skill proposals than
  raw trajectory history.
- **H2:** evolved skills improve unseen-task performance.
- **H3:** keeping knowledge unavailable to the execution agent produces better
  reusable skills.
- **H4:** some skills transfer between models, while others represent
  model-specific compensation.

## 13. Cross-model experiment

Once the basic system works, this is probably the highest-value experiment.
Train skills using Model A, Model B, Model C. Execute every skill on every
model. Produce:

```
                   executor
              A      B      C
skill from A   74     68     71
skill from B   76     81     73
skill from C   72     77     85
```

Then inspect skill contents.
Try to classify rules into:
- environment knowledge
- general procedural knowledge
- tool-specific procedures
- model compensations
- prompt-format hacks

This could become a genuinely interesting research result.

## 14. World simulator

For the first version, keep the environment deterministic and in-process.

```python
world = World(
    inventory={"macbook": 0, "windows": 5},
    employees={"alice": {...}},
    policies={...},
)
```

Expose tools:
- get_employee()
- get_policy()
- get_inventory()
- assign_device()
- create_procurement_request()
- grant_access()
- create_ticket()
- complete_onboarding()

Every tool mutates the same world state.
Reset the world before each evaluation task.
This gives perfect observability and reproducibility.

## 15. Deterministic grading

For each scenario define final-state invariants.

```python
assert world.employee("alice").status != "completed"
assert world.procurement.exists(employee="alice")
assert world.ticket.has_blocker("hardware unavailable")
```

This is substantially better than asking another LLM:
"Did the agent perform well?"
LLM graders can later evaluate softer trajectory qualities.
The primary oracle should remain deterministic.

## 16. CLI

The project should initially be fully operable from the command line.

```
exp run train
exp mine
exp propose onboarding
exp eval candidate-17
exp promote candidate-17
exp evolve --iterations 10
exp compare
exp inspect run-183
```

Useful research command:
```
exp matrix --skill-source '*' --executor-model '*'
```
Output: `results/transfer-matrix.csv`

## 17. Minimal dashboard

Do not build a UI initially.
Generate static HTML or Markdown reports:

```
Iteration 8
Current skill: v4
Training:
18 / 25 success
Observed patterns:
+ hardware inventory not checked
+ access verified too late
+ duplicate ticket creation
Candidate v5:
3 lines modified
Validation:
v4    74%
v5    82%
Regressions:
1
Decision:
ACCEPT
```

This gives enough observability without consuming time on frontend work.

## 18. Technology choices

For a solo implementation I would use:
- Python 3.13
- Pydantic
- Typer
- SQLite
- JSONL
- Git
- pytest
- Jinja2
- LLM API abstraction: LiteLLM or a minimal internal adapter
- Agent: Google ADK only if you specifically want to investigate ADK;
  otherwise start with a tiny custom loop.
  I would actually avoid LangGraph/ADK in version 0.
  You want the experiment to study experience → knowledge → skill evolution,
  not framework semantics.
  A minimal agent loop gives you much tighter experimental control.

## 19. Research instrumentation

Every experiment should emit a manifest:

```yaml
experiment_id: exp-017
model:
  name: gemini-x
  temperature: 0
dataset:
  version: git:283ea1
skill:
  version: git:d291ac
knowledge:
  version: git:c9182a
environment:
  version: git:77128b
seed: 42
```

This will become extremely valuable once you start comparing runs.
Think of every experiment as a reproducible software build.

## 20. Milestones

A realistic progression for a few hours per day is:

- **M0** — deterministic world: scenarios, tools, graders
- **M1** — execution harness: trace capture, active skills
- **M2** — evidence miner: structured knowledge store
- **M3** — skill proposer: patch generation, provenance
- **M4** — validation loop: promotion / rollback
- **M5** — baseline experiments: no skill, trace-to-skill, knowledge-to-skill
- **M6** — cross-model transfer
- **M7** — report / rMax.ai article

Do not add vector databases, embeddings, Kubernetes, MCP, distributed agents,
or sophisticated knowledge graphs before M5.
They create engineering surface without answering the research question.

## 21. Success criteria

The PoC is successful if you can demonstrate a sequence such as:

```
Initial:         54% held-out success
iteration 1:     61%
iteration 2:     candidate rejected
iteration 3:     66%
iteration 4:     68%
iteration 5:     72%
```

while showing exactly:
- which failures generated knowledge
- which knowledge generated a skill change
- which validation evidence justified promotion

That provenance chain is more important than achieving an impressive absolute
score.

## 22. The artifact that matters

The strongest output from the project is not the agent.
It is the evolution record:

```
failure
  ↓
evidence
  ↓
hypothesis
  ↓
procedural modification
  ↓
evaluation
  ↓
accepted knowledge
```

That turns agent improvement into something closer to empirical software
engineering rather than prompt tinkering.

For rMax.ai, position the project as:

> **Experience Compiler Lab — Can agent experience be compiled into validated
> procedural knowledge?**

It sits directly at the intersection of agent memory, skills, harnesses,
provenance and eval-driven improvement, but remains small enough to execute as
a serious solo research project.

The three immediate implementation steps are:
1. build the deterministic enterprise world plus 30–50 scenarios;
2. implement the trace/evaluation harness before adding any LLM-driven
   learning;
3. then add Evidence Miner → Skill Proposer → validation as separate
   components so each hypothesis can be independently ablated.

"If you can't measure it, you can't improve it." — often attributed to Peter Drucker
