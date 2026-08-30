"""evolve() end-to-end tests (SPEC.md §11) plus the held-out split guard.

Isolation: each run builds a tmp tree — datasets/train.jsonl (synthetic dev
scenarios), skills/onboarding/ (v1 skill), results/, experience/, knowledge/.
The execution model is a deterministic double that improves only when the
patched skill text ("improved procedure" marker) reaches the system prompt;
the miner and proposer are FakeModels with inline scripted JSON, the proposer
scripted per iteration (iteration 1 proposes a real improvement, iteration 2
proposes nothing -> the loop stops early).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agent.adapter import CompletionResult, FakeModel, Usage
from experiments.evolution import EvolutionResult, evolve
from traces.schema import Scenario

SKILL_MD = (
    "# Onboarding\n"
    "\n"
    "## Procedure\n"
    "1. Resolve employee identity.\n"
    "2. Assign hardware.\n"
)
PATCH = (
    "@@ Procedure\n"
    "- 2. Assign hardware.\n"
    "+ 2. Assign hardware only if inventory confirms availability (improved procedure).\n"
)
MARKER = "improved procedure"

_PURPOSE = {
    "skill": "onboarding",
    "version": 1,
    "derived_from": [],
    "proposed_by": {"model": "fake"},
    "evaluation": {"previous_score": None, "candidate_score": None},
    "status": "initial",
    "format_version": 1,
}


def _call(tool: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)},
            }
        ],
    }


class _ProcedureModel:
    """Execution double: improves only when the patched skill is active."""

    model = "fake"
    temperature = 0.0

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        system = str(messages[0].get("content", ""))
        task = str(messages[1].get("content", ""))
        step = sum(1 for message in messages if message.get("role") == "assistant")
        response: dict[str, Any] = {"role": "assistant", "content": "done"}
        if MARKER in system and task.startswith("Fully onboard alice"):
            sequence = [
                _call("grant_access", {"employee_id": "alice", "access": "vpn"}, "c1"),
                _call(
                    "assign_device",
                    {"employee_id": "alice", "device_type": "windows"},
                    "c2",
                ),
                _call("complete_onboarding", {"employee_id": "alice"}, "c3"),
                {"role": "assistant", "content": "done"},
            ]
            response = sequence[min(step, len(sequence) - 1)]
        return CompletionResult(
            message=response,
            usage=Usage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.0),
        )


def _run_factory() -> _ProcedureModel:
    return _ProcedureModel()


def _miner_factory() -> FakeModel:
    payload = [
        {
            "kind": "repeated_success_strategy",
            "hypothesis": "Assign hardware only after inventory check",
            "mentioned_tools": [],
            "mentioned_errors": [],
        }
    ]
    response = {"role": "assistant", "content": json.dumps(payload)}
    return FakeModel(
        scripts=[(response, {"input_tokens": 10, "output_tokens": 5})],
        model="fake",
        temperature=0.0,
    )


class _QueuedProposerFactory:
    """Serves one scripted proposer payload per iteration, then 'no patch'."""

    def __init__(self, payloads: list[dict[str, str]]) -> None:
        self._payloads = list(payloads)

    def __call__(self) -> FakeModel:
        payload = (
            self._payloads.pop(0)
            if self._payloads
            else {"reasoning": "nothing worth changing", "patch": ""}
        )
        response = {"role": "assistant", "content": json.dumps(payload)}
        return FakeModel(
            scripts=[(response, {"input_tokens": 10, "output_tokens": 5}) for _ in range(2)],
            model="fake",
            temperature=0.0,
        )


def _scenario(task_id: str, description: str, completed: bool) -> Scenario:
    alice = {"id": "alice", "name": "Alice", "role": "engineer", "department": "eng"}
    return Scenario.model_validate(
        {
            "task_id": task_id,
            "description": description,
            "world": {
                "inventory": {"windows": 1, "macbook": 0},
                "employees": {"alice": alice},
                "policies": {"engineer": {"access_rules": {"engineer": ["vpn"]}}},
            },
            "toolset": ["grant_access", "assign_device", "complete_onboarding"],
            "grader": {
                "success_invariants": [
                    {
                        "path": "employee.alice.status",
                        "op": "==",
                        "value": "completed" if completed else "pending",
                    }
                ]
            },
            "seed": 1,
            "difficulty": 1,
            "version": 1,
        }
    )


def _make_env(root: Path) -> Path:
    """Tmp evolution tree; returns the skills dir."""
    datasets = root / "datasets"
    datasets.mkdir(parents=True)
    dev = [
        _scenario("dev-1", "Fully onboard alice.", completed=True),
        _scenario("dev-2", "Leave alice pending.", completed=False),
    ]
    (datasets / "train.jsonl").write_text(
        "".join(scenario.model_dump_json() + "\n" for scenario in dev), encoding="utf-8"
    )
    skills_dir = root / "skills"
    workflow_dir = skills_dir / "onboarding"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (workflow_dir / "PURPOSE.yaml").write_text(
        yaml.safe_dump(dict(_PURPOSE), sort_keys=True) + "\n", encoding="utf-8"
    )
    return skills_dir


def _evolve(root: Path) -> EvolutionResult:
    skills_dir = _make_env(root)
    proposer = _QueuedProposerFactory(
        [
            {"reasoning": "check inventory before assigning", "patch": PATCH},
            {"reasoning": "nothing worth changing", "patch": ""},
        ]
    )
    return evolve(
        workflow="onboarding",
        iterations=5,  # the empty iteration-2 proposal stops the loop early
        model_factory=_run_factory,
        seed=7,
        validation_scenarios_override=[
            _scenario("val-1", "Fully onboard alice.", completed=True)
        ],
        base_dir=root,
        skills_dir=skills_dir,
        miner_model_factory=_miner_factory,
        proposer_model_factory=proposer,
    )


def test_evolve_two_iterations_end_to_end(tmp_path: Path) -> None:
    result = _evolve(tmp_path)

    assert len(result.iterations) == 2

    first = result.iterations[0]
    assert first.iteration == 1
    assert first.runs_created == 2
    assert first.runs_succeeded == 1  # dev-1 fails under the v1 skill
    assert len(first.new_record_ids) == 1
    assert first.candidate_id == "candidate-01"
    assert first.decision == "accepted"
    assert first.eval is not None
    assert first.eval["previous_score"] == 0.0
    assert first.eval["candidate_score"] == 1.0
    assert first.eval["regressions"] == 0
    assert first.lines_modified == 2

    second = result.iterations[1]
    assert second.iteration == 2
    assert second.runs_succeeded == 2  # the accepted skill now completes dev-1
    assert second.new_record_ids == []  # same evidence, upserted not new
    assert second.candidate_id is None
    assert second.decision is None
    assert second.eval is None

    assert len(result.provenance) == 1
    link = result.provenance[0]
    assert link.candidate_id == "candidate-01"
    assert link.record_ids == first.new_record_ids
    assert link.failure_run_ids  # dev-1's failed run
    assert link.decision == "accepted"

    # The accepted patch deployed through the only mutation path.
    skill_text = (tmp_path / "skills" / "onboarding" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert MARKER in skill_text
    purpose = yaml.safe_load(
        (tmp_path / "skills" / "onboarding" / "PURPOSE.yaml").read_text(encoding="utf-8")
    )
    assert purpose["version"] == 2

    ledger = tmp_path / "results" / "proposal-history.md"
    assert ledger.exists()
    entries = [
        line
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.startswith("| candidate-")
    ]
    assert len(entries) == 1


def test_evolve_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = _evolve(tmp_path / "run1")
    second = _evolve(tmp_path / "run2")
    assert first.model_dump() == second.model_dump()


def test_held_out_split_never_referenced_in_eval_or_evolution() -> None:
    """SPEC.md §10: the held-out test split is M5-only (experiments/compare.py)."""
    repo = Path(__file__).resolve().parents[1]
    sources = sorted((repo / "src" / "evals").glob("*.py"))
    sources.append(repo / "src" / "experiments" / "evolution.py")
    assert sources, "expected eval/evolution sources to exist"
    for path in sources:
        assert "test" + ".jsonl" not in path.read_text(encoding="utf-8"), (
            f"held-out split referenced in {path}"
        )
