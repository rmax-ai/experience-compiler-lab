"""evaluate_candidate tests (SPEC.md §10): determinism, regressions, score vector.

Fixture design: ``SkillReactiveModel`` is a deterministic test double whose
behavior switches on a marker embedded in the skill text inside the system
prompt. ``evaluate_candidate`` builds a fresh model per task via
``model_factory``, so both sides see the IDENTICAL script logic — the only
difference between the baseline and candidate runs is the skill text itself.
That is exactly the fair-comparison contract: same scenarios, same per-task
seeds, same script sequence per side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.adapter import CompletionResult, Usage
from evals.runner import evaluate_candidate
from traces.schema import Scenario
from traces.store import TraceStore

TOOLS = [
    "get_employee",
    "get_policy",
    "get_inventory",
    "assign_device",
    "create_procurement_request",
    "grant_access",
    "create_ticket",
    "complete_onboarding",
]

# Marker strings embedded in the candidate skill text; the reactive model
# improves its procedure when it sees them (baseline text carries no marker).
IMPROVE = "CANDIDATE-IMPROVE"
BREAK = "CANDIDATE-BREAK"

BASELINE_SKILL = "# Onboarding\n\n## Procedure\n1. Do the onboarding.\n"
IMPROVED_SKILL = f"{BASELINE_SKILL}2. {IMPROVE}: verify before acting.\n"
BREAKING_SKILL = f"{BASELINE_SKILL}2. {IMPROVE} and {BREAK}: act eagerly.\n"


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


_FINAL: dict[str, Any] = {"role": "assistant", "content": "done"}


def _plan(system: str, task: str, step: int) -> dict[str, Any]:
    """Deterministic response plan keyed on the skill marker and the task."""
    improved = IMPROVE in system or BREAK in system
    if task.startswith("Task t1"):
        if not improved:
            return _FINAL
        sequence = [
            _call("grant_access", {"employee_id": "alice", "access": "vpn"}, "c1"),
            _call("assign_device", {"employee_id": "alice", "device_type": "windows"}, "c2"),
            _call("complete_onboarding", {"employee_id": "alice"}, "c3"),
            _FINAL,
        ]
        return sequence[min(step, len(sequence) - 1)]
    if task.startswith("Task t2"):
        if BREAK in system:
            sequence = [
                _call("assign_device", {"employee_id": "bob", "device_type": "windows"}, "c1"),
                _FINAL,
            ]
            return sequence[min(step, len(sequence) - 1)]
        return _FINAL
    return _FINAL


class SkillReactiveModel:
    """Same interface as FakeModel; behavior derives from the skill text."""

    model = "fake-reactive"
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
        return CompletionResult(
            message=_plan(system, task, step),
            usage=Usage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.0),
        )


def _factory() -> SkillReactiveModel:
    return SkillReactiveModel()


def _scenario(
    task_id: str,
    description: str,
    employees: dict[str, dict],
    inventory: dict[str, int],
    policies: dict[str, dict],
    success_invariants: list[dict],
    must_not: list[dict] | None = None,
) -> Scenario:
    return Scenario.model_validate(
        {
            "task_id": task_id,
            "description": description,
            "world": {
                "inventory": inventory,
                "employees": employees,
                "policies": policies,
            },
            "toolset": TOOLS,
            "grader": {
                "success_invariants": success_invariants,
                "must_not": must_not or [],
            },
            "seed": 1,
            "difficulty": 1,
            "version": 1,
        }
    )


def _scenarios() -> list[Scenario]:
    """Three tasks: one fixable by the improved skill, one fragile, one stable."""
    alice = {"id": "alice", "name": "Alice", "role": "engineer", "department": "eng"}
    bob = {"id": "bob", "name": "Bob", "role": "engineer", "department": "eng"}
    carol = {"id": "carol", "name": "Carol", "role": "engineer", "department": "eng"}
    policies = {"engineer": {"access_rules": {"engineer": ["vpn"]}}}
    return [
        _scenario(
            "t1-fixable",
            "Task t1: fully onboard alice.",
            {"alice": alice},
            {"windows": 1, "macbook": 0},
            policies,
            [{"path": "employee.alice.status", "op": "==", "value": "completed"}],
        ),
        _scenario(
            "t2-fragile",
            "Task t2: leave bob's hardware untouched.",
            {"bob": bob},
            {"windows": 1, "macbook": 0},
            policies,
            [{"path": "inventory.windows", "op": "==", "value": 1}],
            must_not=[
                {"path": "employee.bob.assigned_device", "op": "==", "value": "windows"}
            ],
        ),
        _scenario(
            "t3-stable",
            "Task t3: nothing to do for carol.",
            {"carol": carol},
            {"windows": 1, "macbook": 0},
            policies,
            [{"path": "inventory.macbook", "op": "==", "value": 0}],
        ),
    ]


def test_evaluate_candidate_deterministic_across_runs(tmp_path: Path) -> None:
    scenarios = _scenarios()
    first = evaluate_candidate(
        BASELINE_SKILL, IMPROVED_SKILL, scenarios, "onboarding", _factory, 42,
        store=TraceStore(tmp_path / "a"),
    )
    second = evaluate_candidate(
        BASELINE_SKILL, IMPROVED_SKILL, scenarios, "onboarding", _factory, 42,
        store=TraceStore(tmp_path / "b"),
    )
    # latency_s is wall-clock and the one nondeterministic metric; everything
    # else must be byte-identical across identical evaluations.
    assert first.model_dump(exclude={"score_vector_delta"}) == second.model_dump(
        exclude={"score_vector_delta"}
    )
    for key, value in first.score_vector_delta.items():
        if key == "latency_s":
            continue
        assert value == second.score_vector_delta[key]


def test_evaluate_candidate_improvement_rates_and_tasks(tmp_path: Path) -> None:
    result = evaluate_candidate(
        BASELINE_SKILL, IMPROVED_SKILL, _scenarios(), "onboarding", _factory, 42,
        store=TraceStore(tmp_path),
    )
    # Baseline: t1 fails (never completed), t2 + t3 pass. Candidate fixes t1.
    assert result.baseline_by_task == {
        "t1-fixable": False,
        "t2-fragile": True,
        "t3-stable": True,
    }
    assert result.candidate_by_task == {
        "t1-fixable": True,
        "t2-fragile": True,
        "t3-stable": True,
    }
    assert result.baseline_success_rate == round(2 / 3, 4)
    assert result.candidate_success_rate == 1.0
    assert result.regressions == []


def test_evaluate_candidate_regressions_detected(tmp_path: Path) -> None:
    result = evaluate_candidate(
        BASELINE_SKILL, BREAKING_SKILL, _scenarios(), "onboarding", _factory, 42,
        store=TraceStore(tmp_path),
    )
    # The breaking candidate fixes t1 but breaks t2 (baseline passed it).
    assert result.regressions == ["t2-fragile"]
    assert result.baseline_success_rate == round(2 / 3, 4)
    assert result.candidate_success_rate == round(2 / 3, 4)


def test_evaluate_candidate_score_vector_tool_calls_delta(tmp_path: Path) -> None:
    result = evaluate_candidate(
        BASELINE_SKILL, IMPROVED_SKILL, _scenarios(), "onboarding", _factory, 42,
        store=TraceStore(tmp_path),
    )
    # Baseline never calls a tool; the candidate uses 3 tool calls on t1.
    assert result.score_vector_delta["tool_calls"] == 3.0
    assert result.score_vector_delta["tokens"] > 0
    assert set(result.score_vector_delta) == {
        "tool_calls",
        "tokens",
        "cost_usd",
        "latency_s",
        "trajectory_length",
    }


def test_evaluate_candidate_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    scenarios = _scenarios()
    scenarios.append(scenarios[0].model_copy())
    with pytest.raises(ValueError, match="duplicate task_id"):
        evaluate_candidate(
            BASELINE_SKILL, IMPROVED_SKILL, scenarios, "onboarding", _factory, 42,
            store=TraceStore(tmp_path),
        )
