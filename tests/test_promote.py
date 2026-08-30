"""promote() tests: accept/reject flows, ledger append-only, protections.

Isolation: each test builds a tmp ``skills/`` tree and a tmp ``base_dir``
(results/, experience/), so the real deployed skill is never touched. The
model is a deterministic double whose behavior improves only when the
patched skill text (carrying the "improved procedure" marker) is in the
system prompt — the fair-comparison seam used by the eval harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent.adapter import CompletionResult, Usage
from experiments.promote import (
    PromotionError,
    evaluate_candidate_record,
    promote,
)
from experiments.proposal_store import ProposalStore
from skills.proposer import Proposal
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
    "version": 3,
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
    """Improves only when the patched skill text is in the system prompt."""

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


def _factory() -> _ProcedureModel:
    return _ProcedureModel()


def _scenario(task_id: str, description: str, completed: bool) -> Scenario:
    alice = {"id": "alice", "name": "Alice", "role": "engineer", "department": "eng"}
    invariant_value = "completed" if completed else "pending"
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
                    {"path": "employee.alice.status", "op": "==", "value": invariant_value}
                ]
            },
            "seed": 1,
            "difficulty": 1,
            "version": 1,
        }
    )


def _make_env(tmp_path: Path) -> Path:
    """Tmp skills tree (onboarding v3) plus candidate store; returns skills dir."""
    skills_dir = tmp_path / "skills"
    workflow_dir = skills_dir / "onboarding"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (workflow_dir / "PURPOSE.yaml").write_text(
        yaml.safe_dump(dict(_PURPOSE), sort_keys=True) + "\n", encoding="utf-8"
    )
    return skills_dir


def _save_candidate(base_dir: Path, candidate_id: str) -> None:
    ProposalStore(base_dir).save_candidate(
        candidate_id=candidate_id,
        proposal=Proposal(reasoning="r", patch=PATCH, cited_records=["rec-a"]),
        workflow="onboarding",
        from_version="3",
        to_version="4",
        proposed_model="fake",
    )


def _ledger_entries(base_dir: Path) -> list[str]:
    path = base_dir / "results" / "proposal-history.md"
    if not path.exists():
        return []
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("| candidate-")
    ]


def test_promote_accepted_flow(tmp_path: Path) -> None:
    skills_dir = _make_env(tmp_path)
    _save_candidate(tmp_path, "candidate-01")
    # Baseline never completes alice (0.0); the patched skill does (1.0).
    evaluate_candidate_record(
        "candidate-01",
        model_factory=_factory,
        base_dir=tmp_path,
        skills_dir=skills_dir,
        validation_scenarios=[_scenario("v1", "Fully onboard alice.", completed=True)],
    )

    outcome = promote("candidate-01", base_dir=tmp_path, skills_dir=skills_dir)

    assert outcome.decision == "accepted"
    skill_text = (skills_dir / "onboarding" / "SKILL.md").read_text(encoding="utf-8")
    assert "improved procedure" in skill_text

    purpose = yaml.safe_load(
        (skills_dir / "onboarding" / "PURPOSE.yaml").read_text(encoding="utf-8")
    )
    assert purpose["version"] == 4
    assert purpose["status"] == "accepted"
    assert purpose["derived_from"] == ["rec-a"]
    assert purpose["evaluation"] == {"previous_score": 0.0, "candidate_score": 1.0}

    record = yaml.safe_load(
        (tmp_path / "results" / "candidates" / "candidate-01" / "record.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert record["decision"] == "accepted"
    assert record["decided_at"] is not None

    proposal_copy = yaml.safe_load(
        (tmp_path / "results" / "proposals" / "candidate-01.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert proposal_copy == record

    entries = _ledger_entries(tmp_path)
    assert len(entries) == 1
    assert "candidate-01" in entries[0]
    assert "ACCEPT" in entries[0]
    assert "v3 -> v4" in entries[0]


def test_promote_rejected_flow_leaves_skills_untouched(tmp_path: Path) -> None:
    skills_dir = _make_env(tmp_path)
    _save_candidate(tmp_path, "candidate-01")
    # A scenario both sides pass untouched: equal rates -> reject.
    evaluate_candidate_record(
        "candidate-01",
        model_factory=_factory,
        base_dir=tmp_path,
        skills_dir=skills_dir,
        validation_scenarios=[_scenario("v1", "Leave alice pending.", completed=False)],
    )

    outcome = promote("candidate-01", base_dir=tmp_path, skills_dir=skills_dir)

    assert outcome.decision == "rejected"
    assert (skills_dir / "onboarding" / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
    purpose = yaml.safe_load(
        (skills_dir / "onboarding" / "PURPOSE.yaml").read_text(encoding="utf-8")
    )
    assert purpose["version"] == 3
    assert purpose["status"] == "initial"

    record = yaml.safe_load(
        (tmp_path / "results" / "candidates" / "candidate-01" / "record.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert record["decision"] == "rejected"
    entries = _ledger_entries(tmp_path)
    assert len(entries) == 1
    assert "REJECT" in entries[0]


def test_promote_double_promote_raises(tmp_path: Path) -> None:
    skills_dir = _make_env(tmp_path)
    _save_candidate(tmp_path, "candidate-01")
    evaluate_candidate_record(
        "candidate-01",
        model_factory=_factory,
        base_dir=tmp_path,
        skills_dir=skills_dir,
        validation_scenarios=[_scenario("v1", "Fully onboard alice.", completed=True)],
    )
    promote("candidate-01", base_dir=tmp_path, skills_dir=skills_dir)
    with pytest.raises(PromotionError, match="already decided"):
        promote("candidate-01", base_dir=tmp_path, skills_dir=skills_dir)


def test_promote_unevaluated_raises(tmp_path: Path) -> None:
    skills_dir = _make_env(tmp_path)
    _save_candidate(tmp_path, "candidate-01")
    with pytest.raises(PromotionError, match="has not been evaluated"):
        promote("candidate-01", base_dir=tmp_path, skills_dir=skills_dir)


def test_ledger_is_append_only_across_decisions(tmp_path: Path) -> None:
    skills_dir = _make_env(tmp_path)
    _save_candidate(tmp_path, "candidate-01")
    _save_candidate(tmp_path, "candidate-02")

    # Both candidates are evaluated against the SAME pre-promotion skill
    # (evaluation applies each patch to the current skill, so it must happen
    # before candidate-01's accept mutates skills/).
    evaluate_candidate_record(
        "candidate-01",
        model_factory=_factory,
        base_dir=tmp_path,
        skills_dir=skills_dir,
        validation_scenarios=[_scenario("v1", "Fully onboard alice.", completed=True)],
    )
    evaluate_candidate_record(
        "candidate-02",
        model_factory=_factory,
        base_dir=tmp_path,
        skills_dir=skills_dir,
        validation_scenarios=[_scenario("v1", "Leave alice pending.", completed=False)],
    )

    promote("candidate-01", base_dir=tmp_path, skills_dir=skills_dir)
    ledger_path = tmp_path / "results" / "proposal-history.md"
    after_first = ledger_path.read_bytes()

    promote("candidate-02", base_dir=tmp_path, skills_dir=skills_dir)

    after_second = ledger_path.read_bytes()
    # The second decision was appended; every prior byte is untouched.
    assert after_second.startswith(after_first)
    assert len(after_second) > len(after_first)
    assert len(_ledger_entries(tmp_path)) == 2
