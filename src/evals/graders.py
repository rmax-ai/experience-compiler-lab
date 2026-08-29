"""Deterministic final-state graders (SPEC.md §15, docs/data-formats.md §1).

Pure functions over a :class:`world.state.World` and a scenario ``grader``
spec: no mutation, deterministic ordering of reported violations.

Semantics
---------
- ``success_invariants``: all must hold for success.
- ``constraint_invariants``: failures are recorded in ``violated_constraints``
  and do NOT flip success.
- ``must_not``: failures are recorded in ``errors`` AND flip success.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from traces.schema import GraderSpec
from world.state import World

# Path syntax: employee.<id>.<field>, inventory.<type>,
# procurement.exists(employee=<id>), ticket.has_blocker(<text>).
_EMPLOYEE_PATH = re.compile(r"^employee\.(?P<employee_id>[^.\s]+)\.(?P<field>[^.\s]+)$")
_INVENTORY_PATH = re.compile(r"^inventory\.(?P<device>[^.\s]+)$")
_PROCUREMENT_EXISTS = re.compile(
    r"^procurement\.exists\(\s*employee\s*=\s*(?P<employee>[^)\s]+)\s*\)$"
)
_TICKET_HAS_BLOCKER = re.compile(
    r"^ticket\.has_blocker\(\s*(?P<text>[^)]*)\s*\)$"
)


class GraderResult(BaseModel):
    """Outcome of evaluating one scenario's grader against a world state."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    errors: list[str]
    violated_constraints: list[str]


def evaluate(world: World, grader_spec: GraderSpec | dict[str, Any]) -> GraderResult:
    """Evaluate final-state invariants against ``world``.

    ``grader_spec`` is the scenario ``grader`` block: either a
    :class:`GraderSpec` model or the raw dict from a scenario file (it is
    coerced via :class:`GraderSpec`).
    """
    if isinstance(grader_spec, dict):
        grader_spec = GraderSpec.model_validate(grader_spec)

    errors: list[str] = []
    violated: list[str] = []
    success = True

    for invariant in grader_spec.success_invariants:
        actual = _resolve(world, invariant.path)
        if not _holds(actual, invariant.op, invariant.value):
            success = False
            errors.append(_failure(invariant.path, invariant.op, invariant.value, actual))

    for invariant in grader_spec.constraint_invariants:
        actual = _resolve(world, invariant.path)
        if not _holds(actual, invariant.op, invariant.value):
            violated.append(_failure(invariant.path, invariant.op, invariant.value, actual))

    for invariant in grader_spec.must_not:
        actual = _resolve(world, invariant.path)
        if _holds(actual, invariant.op, invariant.value):
            success = False
            errors.append(f"must_not violated: {invariant.path} {invariant.op} {invariant.value}")

    return GraderResult(
        success=success,
        errors=sorted(errors),
        violated_constraints=sorted(violated),
    )


def _resolve(world: World, path: str) -> Any:
    """Resolve a path expression to a value in the world (or ``None``)."""
    if (match := _EMPLOYEE_PATH.match(path)) is not None:
        employee = world.employees.get(match.group("employee_id"))
        if employee is None:
            return None
        return _employee_field(employee, match.group("field"))

    if (match := _INVENTORY_PATH.match(path)) is not None:
        device = match.group("device")
        return world.inventory.get(device, 0)

    if (match := _PROCUREMENT_EXISTS.match(path)) is not None:
        employee = match.group("employee")
        return any(request.employee == employee for request in world.procurement)

    if (match := _TICKET_HAS_BLOCKER.match(path)) is not None:
        text = match.group("text")
        return any(
            ticket.blocker is not None and text in ticket.blocker for ticket in world.tickets
        )

    return None


def _employee_field(employee: Any, field: str) -> Any:
    if field == "assigned_device":
        return employee.assigned_device.value if employee.assigned_device is not None else None
    if field == "granted_access":
        return list(employee.granted_access)
    return getattr(employee, field, None)


def _holds(actual: Any, op: str, expected: Any) -> bool:
    if op == "exists":
        return bool(actual)
    if actual is None:
        return False
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        return isinstance(expected, str) and expected in actual
    return False


def _failure(path: str, op: str, expected: Any, actual: Any) -> str:
    if op == "exists":
        return f"invariant failed: {path} {op} (actual: {actual!r})"
    return f"invariant failed: {path} {op} {expected!r} (actual: {actual!r})"
