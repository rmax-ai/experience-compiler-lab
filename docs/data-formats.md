# Data Formats — contracts shared by all components

Versioned contracts. Changing any of these = breaking change: bump the
`version` field, update this doc FIRST, then code.

## 1. Scenario (datasets/*.jsonl, one JSON object per line)

```json
{
  "task_id": "onboard_alice_basic",
  "description": "Onboard employee Alice.",
  "world": {
    "inventory": {"macbook": 0, "windows": 5},
    "employees": {"alice": {"id": "alice", "name": "Alice", "role": "engineer", "status": "pending"}},
    "policies": {"access": "engineers require vpn + github"}
  },
  "toolset": ["get_employee", "get_policy", "get_inventory", "assign_device",
              "create_procurement_request", "grant_access", "create_ticket",
              "complete_onboarding"],
  "grader": {
    "success_invariants": [
      {"path": "employee.alice.status", "op": "==", "value": "completed"}
    ],
    "constraint_invariants": [
      {"path": "procurement.exists(employee=alice)", "op": "==", "value": true},
      {"path": "ticket.has_blocker(hardware unavailable)", "op": "==", "value": true}
    ],
    "must_not": [
      {"path": "employee.alice.assigned_device", "op": "==", "value": "macbook"}
    ]
  },
  "seed": 42,
  "difficulty": 1,
  "version": 1
}
```

- `success_invariants`: all must hold for `Outcome.success == true`.
- `constraint_invariants`: recorded as `violated_constraints` when they fail;
  they do NOT flip success. Used for the score vector.
- `must_not`: hard violations — flip success to false.
- Graders evaluate against the world's final state after the agent finishes
  (or after step budget exhaustion). `op` ∈ `==`, `!=`, `in`, `exists`.
- World state resets from the scenario's `world` block before every run.

## 2. Trace (experience/runs/<run_id>.jsonl + manifest)

Pydantic models in `src/traces/schema.py` (SPEC.md §5):

```
Trace:    run_id, task_id, model, skill_version, messages[], actions[],
          final_answer, outcome, metrics, manifest
Action:   index, tool, arguments, result, timestamp, world_state_before,
          world_state_after
Outcome:  success, errors[], violated_constraints[]
Metrics:  tool_calls, tokens_in, tokens_out, estimated_cost_usd, latency_s,
          recovery_count, trajectory_length
```

`world_state_*` are compact deterministic snapshots (sorted key=value lines),
not full object dumps. Manifest (SPEC.md §19): experiment_id, model name +
temperature, seed, and git versions of dataset/skill/knowledge/environment.

Immutable after run close — a run file is never rewritten.

## 3. Knowledge record (knowledge/patterns/<id>.yaml, indexed by index.yaml)

```yaml
id: inventory-check-before-assignment
claim:
  type: procedure
  text: Check available inventory before assigning hardware.
scope:
  workflows: [onboarding]
evidence:
  supporting_runs: [run_014, run_031]
  counterexamples: [run_022]
statistics: {support: 8, failures: 2}
confidence: 0.82
status: active            # active | superseded
first_seen: 2026-08-29
last_updated: 2026-08-29
version: 1
```

Append-only: updates bump `last_updated` and `statistics`; a record is
retired via `status: superseded` + `supersedes`/`superseded_by` references —
never deleted or overwritten in place.

## 4. Skill (skills/<workflow>/SKILL.md + PURPOSE.yaml)

```markdown
# Employee Onboarding
## Objective
Complete onboarding while respecting workflow dependencies.
## Procedure
1. Resolve employee identity.
2. Retrieve onboarding requirements.
3. Check hardware inventory.
...
## Failure handling
If a required dependency cannot be fulfilled:
- record the blocker;
- escalate to the responsible team;
- leave the workflow incomplete.
```

```yaml
skill: onboarding
version: 4
derived_from: [inventory-check-before-assignment, incomplete-access-before-close]
proposed_by: {model: claude-x}
evaluation: {previous_score: 0.71, candidate_score: 0.79}
status: accepted           # accepted | rejected
version: 1
```

Every skill change is a patch (diff format below) applied to SKILL.md with a
PURPOSE.yaml update. Rejected patches stay in the proposal history forever.

## 5. Patch (candidate artifacts, results/candidates/<id>.diff)

Unified-diff-style patch restricted to skill files:

```diff
@@ Procedure
- 3. Assign required hardware.
+ 3. Check hardware inventory.
+ 4. Assign hardware only if inventory confirms availability.
+ 5. Otherwise create a procurement blocker and escalate.
```

Strict grammar: only `@@ <section>` headers and `-`/`+` lines. The proposer
emits patches; `skills/patch.py` applies and verifies them deterministically
(patch must apply cleanly to the current SKILL.md or it is rejected at
proposal time).

## 6. Proposal record (results/proposals/<candidate-id>.yaml)

```yaml
candidate_id: candidate-17
skill: onboarding
from_version: 4
to_version: 5
diff_file: results/candidates/candidate-17.diff
evidence_refs: [inventory-check-before-assignment]
evaluation: {previous_score: 0.71, candidate_score: 0.79, regressions: 1,
             score_vector: {tool_calls_delta: +2, tokens_delta: -120}}
decision: accepted        # accepted | rejected
decided_at: 2026-08-29T14:03:00Z
version: 1
```
