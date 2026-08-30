"""Tests for optional execution-time memory notes."""

from pathlib import Path

from agent.context import build_context
from traces.schema import Scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_task() -> Scenario:
    return load_scenarios(str(REPO_ROOT / "datasets" / "validation.jsonl"))[0]


def test_build_context_omits_memory_block_without_notes() -> None:
    context = build_context(_load_task())

    assert "Memory notes" not in context.system


def test_build_context_adds_memory_notes_without_changing_other_fields() -> None:
    task = _load_task()
    without_notes = build_context(task)
    with_notes = build_context(
        task,
        memory_notes=["Check available inventory before assigning hardware."],
    )

    assert "## Memory notes" in with_notes.system
    assert "- Check available inventory before assigning hardware." in with_notes.system
    assert with_notes.skill == without_notes.skill
    assert with_notes.tools == without_notes.tools
    assert with_notes.task == without_notes.task
