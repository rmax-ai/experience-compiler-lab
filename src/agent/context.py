"""Execution-time context assembly (PYTHON_ARCHITECTURE.md "Context assembly").

The context handed to the executor contains exactly four inputs: system
instructions, the active skill markdown, the task description, and the tool
schemas. It intentionally carries no knowledge content — the H3 mechanism.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from agent.system_prompt import SYSTEM_INSTRUCTIONS
from skills.loader import load_skill
from traces.schema import Scenario
from world.api import TOOL_SCHEMAS


class ExecutionContext(BaseModel):
    """Everything the execution agent is allowed to see (H3)."""

    model_config = ConfigDict(extra="forbid")

    system: str
    skill: str
    task: str
    tools: list[dict]


def build_context(
    task: Scenario,
    workflow: str = "onboarding",
    skill_loader: Callable[[str], str] | None = None,
) -> ExecutionContext:
    """Assemble the executor context for one task.

    The skill text is loaded from ``skills/<workflow>/SKILL.md`` (via
    ``skills.loader`` unless a ``skill_loader`` is injected for tests). Tools
    come from ``world.TOOL_SCHEMAS`` — all 8 schemas, no implementation hints.
    """
    loader = skill_loader or load_skill
    return ExecutionContext(
        system=SYSTEM_INSTRUCTIONS,
        skill=loader(workflow),
        task=task.description,
        tools=[dict(schema) for schema in TOOL_SCHEMAS],
    )
