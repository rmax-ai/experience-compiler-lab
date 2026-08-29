"""Executor tests: scripted success/failure, budget exhaustion, H3 separation."""

import json
from pathlib import Path

from agent.adapter import FakeModel
from agent.context import build_context
from agent.executor import Executor, build_world
from experiments.runner import run_tasks
from traces.schema import Manifest, Scenario, load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_task(task_id: str, split: str = "validation") -> Scenario:
    scenarios = load_scenarios(str(REPO_ROOT / "datasets" / f"{split}.jsonl"))
    for scenario in scenarios:
        if scenario.task_id == task_id:
            return scenario
    raise AssertionError(f"scenario not found: {task_id}")


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, sort_keys=True),
                },
            }
        ],
    }


def _answer(text: str) -> dict:
    return {"role": "assistant", "content": text}


_USAGE = {"input_tokens": 100, "output_tokens": 40}


def _factory(scripts: list[tuple[dict, dict]]) -> object:
    """model_factory returning a fresh FakeModel per task (isolated script state)."""

    def factory() -> FakeModel:
        return FakeModel(scripts=[(dict(response), dict(usage)) for response, usage in scripts])

    return factory


def test_scripted_success_path(tmp_path: Path) -> None:
    scenario = _load_task("onboard_alice_basic_2")
    scripts = [
        (_tool_call("get_employee", {"employee_id": "alice"}, "c1"), _USAGE),
        (_tool_call("get_policy", {"role": "engineer"}, "c2"), _USAGE),
        (_tool_call("get_inventory", {}, "c3"), _USAGE),
        (_tool_call("assign_device", {"employee_id": "alice", "device_type": "macbook"}, "c4"), _USAGE),
        (_tool_call("grant_access", {"employee_id": "alice", "access": "vpn"}, "c5"), _USAGE),
        (_tool_call("grant_access", {"employee_id": "alice", "access": "github"}, "c6"), _USAGE),
        (_tool_call("complete_onboarding", {"employee_id": "alice"}, "c7"), _USAGE),
        (_answer("Onboarding complete for Alice."), _USAGE),
    ]
    traces = run_tasks(
        scenarios=[scenario],
        workflow="onboarding",
        model_factory=_factory(scripts),
        seed=42,
        experiment_id="test-success",
        store=TraceStore(tmp_path),
    )
    trace = traces[0]
    assert len(trace.actions) == 7
    assert trace.metrics.tool_calls == len(trace.actions)
    for action in trace.actions:
        assert action.world_state_before, "world_state_before snapshot missing"
        assert action.world_state_after, "world_state_after snapshot missing"
        assert action.tool and action.result is not None
    assert trace.outcome.success is True
    assert trace.final_answer == "Onboarding complete for Alice."


def test_scripted_failure_path_with_recovery(tmp_path: Path) -> None:
    """macbook is out of stock: first assign fails, a later retry on windows succeeds."""
    scenario = _load_task("shortage_alice_macbook_3")
    scripts = [
        (_tool_call("get_employee", {"employee_id": "alice"}, "c1"), _USAGE),
        (_tool_call("get_inventory", {}, "c2"), _USAGE),
        (_tool_call("assign_device", {"employee_id": "alice", "device_type": "macbook"}, "c3"), _USAGE),
        (_tool_call("assign_device", {"employee_id": "alice", "device_type": "windows"}, "c4"), _USAGE),
        (_tool_call("grant_access", {"employee_id": "alice", "access": "vpn"}, "c5"), _USAGE),
        (_tool_call("grant_access", {"employee_id": "alice", "access": "github"}, "c6"), _USAGE),
        (_answer("Device assigned; procurement and escalation pending."), _USAGE),
    ]
    traces = run_tasks(
        scenarios=[scenario],
        workflow="onboarding",
        model_factory=_factory(scripts),
        seed=42,
        experiment_id="test-failure",
        store=TraceStore(tmp_path),
    )
    trace = traces[0]

    failed = [a for a in trace.actions if a.result.get("ok") is False]
    assert len(failed) == 1
    assert failed[0].tool == "assign_device"
    assert "inventory error" in failed[0].result["error"]

    # a later assign_device (windows) retries the same tool successfully
    assert trace.metrics.recovery_count == 1

    # grader errors surface in the outcome (onboarding never completed)
    assert trace.outcome.success is False
    assert any("employee.alice.status" in error for error in trace.outcome.errors)


def test_step_budget_exhaustion() -> None:
    scenario = _load_task("onboard_alice_basic_2")
    world = build_world(scenario)
    scripts = [
        (_tool_call("get_employee", {"employee_id": "alice"}, "c1"), _USAGE),
        (_tool_call("get_policy", {"role": "engineer"}, "c2"), _USAGE),
    ]
    model = FakeModel(scripts=[(dict(r), dict(u)) for r, u in scripts])
    executor = Executor(model=model, world=world, max_steps=2)
    context = build_context(scenario)
    manifest = Manifest(
        experiment_id="test-budget",
        model={"name": "fake", "temperature": 0.0},
        dataset_version="a",
        skill_version="b",
        knowledge_version="c",
        environment_version="d",
        seed=42,
    )
    trace = executor.run(
        task=scenario,
        context=context,
        seed=42,
        run_id="run_999",
        manifest=manifest,
    )
    assert trace.final_answer is None
    assert "step budget exhausted" in trace.outcome.errors
    assert trace.outcome.success is False
    assert trace.metrics.tool_calls == 2


def test_agent_and_experiments_never_import_knowledge() -> None:
    """H3 in code: the execution path never imports the knowledge package."""
    for module_dir in ("src/agent", "src/experiments"):
        for path in sorted((REPO_ROOT / module_dir).glob("*.py")):
            source = path.read_text(encoding="utf-8")
            assert "import knowledge" not in source, f"H3 violation in {path}"
            assert "from knowledge" not in source, f"H3 violation in {path}"
