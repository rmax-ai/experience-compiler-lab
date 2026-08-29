"""ExecutionContext tests: skill text, tool schemas, task, no knowledge content."""

from pathlib import Path

from agent.context import ExecutionContext, build_context
from traces.schema import Scenario, load_scenarios
from world.api import TOOL_SCHEMAS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_task(task_id: str = "onboard_alice_basic_2") -> Scenario:
    scenarios = load_scenarios(str(REPO_ROOT / "datasets" / "validation.jsonl"))
    for scenario in scenarios:
        if scenario.task_id == task_id:
            return scenario
    raise AssertionError(f"scenario not found: {task_id}")


def test_build_context_includes_skill_tools_and_task() -> None:
    scenario = _load_task()
    context = build_context(scenario, workflow="onboarding")

    skill_text = (REPO_ROOT / "skills" / "onboarding" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert context.skill == skill_text
    assert "Procedure" in context.skill

    assert isinstance(context, ExecutionContext)
    assert len(context.tools) == 8
    assert [tool["name"] for tool in context.tools] == [
        schema["name"] for schema in TOOL_SCHEMAS
    ]
    assert context.task == scenario.description


def test_build_context_accepts_injected_skill_loader() -> None:
    scenario = _load_task()
    injected = "## Injected skill\n1. Do the thing."
    context = build_context(
        scenario, workflow="onboarding", skill_loader=lambda workflow: injected
    )
    assert context.skill == injected


def test_context_has_no_knowledge_content() -> None:
    """H3: the execution context never references knowledge paths or content."""
    scenario = _load_task()
    context = build_context(scenario)
    for value in context.model_dump().values():
        assert "knowledge" not in str(value)
