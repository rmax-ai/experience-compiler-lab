# ROADMAP.md — experience-compiler-lab

Milestones map 1:1 to SPEC.md §20. Story numbers refer to the GitHub board.

## v0.1.0 — the PoC (stories M0–M5)

- [ ] **M0 — deterministic world** (story M0): World API (documents, tickets,
      inventory, users, workflows), 8 tools, 50 scenarios (30/10/10 split),
      deterministic graders, world reset per task.
- [ ] **M1 — execution harness** (story M1): tiny agent loop, trace capture
      (Trace/Action/Outcome/Metrics per docs/data-formats.md), `exp run`,
      trace store (JSONL + SQLite index), active skill injection.
- [ ] **M2 — evidence miner** (story M2): candidate extraction prompt,
      deterministic merger, structured knowledge store with evidence links,
      `exp mine`, `knowledge/index.yaml`.
- [ ] **M3 — skill proposer** (story M3): diff-style patch generation,
      PURPOSE.yaml provenance, `exp propose`, `exp inspect`.
- [ ] **M4 — validation loop** (story M4): fixed validation set eval,
      promotion policy, permanent proposal history, `exp eval` / `exp
      promote` / `exp evolve --iterations N`, Markdown iteration reports.
- [ ] **M5 — baseline experiments** (story M5): four configurations
      (Baseline, Trace2Skill, Memory Agent, Experience Compiler) as one
      ablation switch; `exp compare`; manifest per experiment.

Acceptance: provenance chain for ≥1 accepted and ≥1 rejected candidate;
held-out success numbers per configuration; all gates green.

## v0.2.0 — cross-model transfer (story M6)

- [ ] Train skills on models A/B/C; execute every skill on every model.
- [ ] `exp matrix --skill-source '*' --executor-model '*'` →
      `results/transfer-matrix.csv`.
- [ ] Rule classification: environment vs procedural vs tool-specific vs
      model compensation vs prompt-format hack (SPEC.md §13).

## Later / out of scope for PoC

- M7 — rMax.ai article (owned by the publishing pipeline, not this repo).
- Semantic skill retrieval, general-purpose memory, vector search,
  distributed execution, online continuous learning (SPEC.md §1 explicitly
  defers these).
