"""Scenario contract for the task datasets (docs/data-formats.md §1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Literal fields allow pydantic to validate the docs' allowed values but still
# carry the literal text (e.g. "employee.alice.assigned_device").
OpType = Literal["==", "!=", "in", "exists"]


class WorldInit(BaseModel):
    """Initial world state block of a scenario."""

    model_config = ConfigDict(extra="forbid")

    inventory: dict[str, int] = Field(default_factory=dict)
    employees: dict[str, dict] = Field(default_factory=dict)
    policies: dict[str, dict] = Field(default_factory=dict)
    documents: dict[str, dict] = Field(default_factory=dict)
    workflows: dict[str, list[str]] = Field(default_factory=dict)


class Invariant(BaseModel):
    """One final-state invariant checked by the deterministic grader."""

    model_config = ConfigDict(extra="forbid")

    path: str
    op: OpType
    value: bool | str | int | None = None


class GraderSpec(BaseModel):
    """Grader configuration for a scenario."""

    model_config = ConfigDict(extra="forbid")

    success_invariants: list[Invariant] = Field(default_factory=list)
    constraint_invariants: list[Invariant] = Field(default_factory=list)
    must_not: list[Invariant] = Field(default_factory=list)


class Scenario(BaseModel):
    """One task dataset row (docs/data-formats.md §1)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    description: str
    world: WorldInit
    toolset: list[str]
    grader: GraderSpec
    seed: int
    difficulty: int = Field(ge=1, le=3)
    version: int = 1


def load_scenarios(path: str) -> list[Scenario]:
    """Load a scenario JSONL file, one Scenario per line.

    Raises a ValueError naming the offending line number (1-based) on any
    parse or validation failure.
    """
    scenarios: list[Scenario] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:  # tolerate trailing blank lines
                continue
            try:
                scenarios.append(Scenario.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(f"invalid scenario at line {lineno} of {path}: {exc}") from exc
    return scenarios
