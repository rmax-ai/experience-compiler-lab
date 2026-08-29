"""Trace contract (docs/data-formats.md §2) and Scenario contract (§1).

Trace/Action/Outcome/Metrics/Message/Manifest are the immutable run artifact
written by ``experiments.runner`` via ``traces.store.TraceStore``. All models
forbid extra fields so schema drift fails loudly at IO boundaries.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Any, Literal

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


def git_short_hash() -> str:
    """Git short hash of HEAD at run start (manifest versions, SPEC.md §19).

    Falls back to ``"uncommitted"`` when git is unavailable or fails, so runs
    stay reproducible even outside a checkout.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
    if proc.returncode != 0:
        return "uncommitted"
    short = proc.stdout.strip()
    return short or "uncommitted"


class Message(BaseModel):
    """One chat-completions-style message in the run transcript."""

    model_config = ConfigDict(extra="forbid")

    role: str  # system | user | assistant | tool
    content: str
    tool_calls: list[dict] | None = None


class Action(BaseModel):
    """One tool invocation with the world state around it (docs §2)."""

    model_config = ConfigDict(extra="forbid")

    index: int
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    timestamp: datetime  # timezone-aware, UTC
    world_state_before: str  # compact deterministic snapshot
    world_state_after: str


class Outcome(BaseModel):
    """Final run outcome; ``success`` is decided by the grader (docs §1/§2)."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    errors: list[str] = Field(default_factory=list)
    violated_constraints: list[str] = Field(default_factory=list)


class Metrics(BaseModel):
    """Run-level instrumentation (docs §2, SPEC.md §10 score vector)."""

    model_config = ConfigDict(extra="forbid")

    tool_calls: int
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    latency_s: float
    recovery_count: int
    trajectory_length: int


class Manifest(BaseModel):
    """Reproducibility manifest (SPEC.md §19); versions are git short hashes."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    model: dict[str, Any]  # {"name": str, "temperature": float}
    dataset_version: str
    skill_version: str
    knowledge_version: str
    environment_version: str
    seed: int


class Trace(BaseModel):
    """Immutable execution trace (SPEC.md §5, docs/data-formats.md §2).

    Written once by the runner to ``experience/runs/<run_id>.jsonl`` and
    never rewritten. ``model_dump_json``/``model_validate_json`` must be a
    lossless roundtrip.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    model: str
    skill_version: str
    messages: list[Message]
    actions: list[Action]
    final_answer: str | None = None
    outcome: Outcome
    metrics: Metrics
    manifest: Manifest


def to_jsonl(trace: Trace) -> str:
    """Serialize a trace as one JSONL line (no trailing newline)."""
    return trace.model_dump_json()


def from_jsonl(line: str) -> Trace:
    """Parse one JSONL line back into a Trace."""
    return Trace.model_validate_json(line)
