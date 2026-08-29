"""Deterministic in-process enterprise world (SPEC.md §14).

Pydantic v2 models only. The world is the single mutable state object the
tools in :mod:`world.api` operate on; ``snapshot()`` and ``reset()`` make it
safe to reuse across evaluation tasks (world_state_before/after in traces,
M1).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now (never ``datetime.utcnow()``)."""
    return datetime.now(timezone.utc)  # noqa: UP017 — AGENTS.md mandates timezone.utc


class DeviceType(StrEnum):
    """Hardware device types tracked in the inventory."""

    MACBOOK = "macbook"
    WINDOWS = "windows"


class Employee(BaseModel):
    """A synthetic enterprise employee being onboarded."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    role: str
    department: str
    status: str = "pending"  # pending | in_progress | completed
    assigned_device: DeviceType | None = None
    granted_access: list[str] = Field(default_factory=list)


class Ticket(BaseModel):
    """An escalation / tracking ticket. A set ``blocker`` marks an escalation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    status: str = "open"  # open | blocked | resolved
    blocker: str | None = None
    related_employee: str | None = None


class ProcurementRequest(BaseModel):
    """An open purchase request for a device type."""

    model_config = ConfigDict(extra="forbid")

    id: str
    employee: str
    device_type: DeviceType
    status: str = "open"  # open | fulfilled


class Policy(BaseModel):
    """Access policy: role -> required access grants (e.g. engineer -> [vpn, github])."""

    model_config = ConfigDict(extra="forbid")

    access_rules: dict[str, list[str]] = Field(default_factory=dict)


class Document(BaseModel):
    """A retrievable document, e.g. onboarding policy text."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    content: str


class World(BaseModel):
    """The whole deterministic world state.

    ``inventory`` is keyed by :class:`DeviceType` value (``"macbook"`` /
    ``"windows"``). ``workflows`` maps workflow name -> ordered step names.
    """

    model_config = ConfigDict(extra="forbid")

    inventory: dict[str, int] = Field(default_factory=dict)
    employees: dict[str, Employee] = Field(default_factory=dict)
    tickets: list[Ticket] = Field(default_factory=list)
    procurement: list[ProcurementRequest] = Field(default_factory=list)
    policies: dict[str, Policy] = Field(default_factory=dict)
    documents: dict[str, Document] = Field(default_factory=dict)
    workflows: dict[str, list[str]] = Field(default_factory=dict)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        # Deep copy of the constructor state; reset() restores exactly this.
        self._initial = copy.deepcopy(self)

    def snapshot(self) -> str:
        """Deterministic compact snapshot: sorted ``key=value`` lines per section.

        Two identical worlds produce byte-identical snapshots. This is the
        ``world_state_before``/``world_state_after`` format for traces (M1).
        """
        sections: list[str] = []

        inventory_lines = [
            f"{key}={value}" for key, value in sorted(self.inventory.items())
        ]
        if inventory_lines:
            sections.append("inventory:" + "\n".join(inventory_lines))

        if self.employees:
            sections.append("employees:" + "\n".join(sorted(self._employee_lines())))

        if self.tickets:
            sections.append("tickets:" + "\n".join(sorted(self._ticket_lines())))

        if self.procurement:
            sections.append(
                "procurement:" + "\n".join(sorted(self._procurement_lines()))
            )

        if self.policies:
            policy_lines = []
            for role in sorted(self.policies):
                rules = self.policies[role].access_rules.get(role, [])
                policy_lines.append(f"{role}=" + ",".join(sorted(rules)))
            sections.append("policies:" + "\n".join(policy_lines))

        if self.documents:
            doc_lines = [
                f"{doc_id}={doc.title}" for doc_id, doc in sorted(self.documents.items())
            ]
            sections.append("documents:" + "\n".join(doc_lines))

        workflow_lines = [
            f"{name}=" + ",".join(steps)
            for name, steps in sorted(self.workflows.items())
        ]
        if workflow_lines:
            sections.append("workflows:" + "\n".join(workflow_lines))

        return "\n".join(sections)

    def reset(self) -> None:
        """Restore the exact state this world was constructed with."""
        current: World = copy.deepcopy(self._initial)
        self.inventory = current.inventory
        self.employees = current.employees
        self.tickets = current.tickets
        self.procurement = current.procurement
        self.policies = current.policies
        self.documents = current.documents
        self.workflows = current.workflows
        self._initial = copy.deepcopy(self)

    def _employee_lines(self) -> list[str]:
        lines: list[str] = []
        for emp in sorted(self.employees.values(), key=lambda e: e.id):
            device = emp.assigned_device.value if emp.assigned_device else "-"
            access = ",".join(sorted(emp.granted_access)) or "-"
            lines.append(
                f"{emp.id}.name={emp.name};"
                f"{emp.id}.role={emp.role};"
                f"{emp.id}.department={emp.department};"
                f"{emp.id}.status={emp.status};"
                f"{emp.id}.assigned_device={device};"
                f"{emp.id}.granted_access={access}"
            )
        return lines

    def _ticket_lines(self) -> list[str]:
        lines: list[str] = []
        for t in sorted(self.tickets, key=lambda t: t.id):
            blocker = t.blocker if t.blocker is not None else "-"
            related = t.related_employee if t.related_employee is not None else "-"
            lines.append(
                f"{t.id}.title={t.title};{t.id}.status={t.status};"
                f"{t.id}.blocker={blocker};{t.id}.related_employee={related}"
            )
        return lines

    def _procurement_lines(self) -> list[str]:
        lines: list[str] = []
        for p in sorted(self.procurement, key=lambda p: p.id):
            lines.append(
                f"{p.id}.employee={p.employee};{p.id}.device_type={p.device_type.value};"
                f"{p.id}.status={p.status}"
            )
        return lines
