"""Eight deterministic tools operating on one :class:`world.state.World`.

Every tool is callable with keyword arguments and returns a dict result:
``{"ok": bool, "error": str | None, "data": dict | None}``. Failures MUST
return ``ok=false`` with a clear error string, never raise.

``TOOL_SCHEMAS`` is the JSON-schema contract the execution agent sees (M1
uses it for function calling); it lists the same 8 tools.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from world.state import DeviceType, ProcurementRequest, Ticket, World

# Tool name constants — single source of truth for both the API and schemas.
TOOL_GET_EMPLOYEE = "get_employee"
TOOL_GET_POLICY = "get_policy"
TOOL_GET_INVENTORY = "get_inventory"
TOOL_ASSIGN_DEVICE = "assign_device"
TOOL_CREATE_PROCUREMENT_REQUEST = "create_procurement_request"
TOOL_GRANT_ACCESS = "grant_access"
TOOL_CREATE_TICKET = "create_ticket"
TOOL_COMPLETE_ONBOARDING = "complete_onboarding"

TOOL_NAMES: list[str] = [
    TOOL_GET_EMPLOYEE,
    TOOL_GET_POLICY,
    TOOL_GET_INVENTORY,
    TOOL_ASSIGN_DEVICE,
    TOOL_CREATE_PROCUREMENT_REQUEST,
    TOOL_GRANT_ACCESS,
    TOOL_CREATE_TICKET,
    TOOL_COMPLETE_ONBOARDING,
]

Result = dict[str, Any]

_OK: Result = {"ok": True, "error": None, "data": None}


def _ok(**data: Any) -> Result:
    return {"ok": True, "error": None, "data": data or None}


def _err(message: str) -> Result:
    return {"ok": False, "error": message, "data": None}


def _resolve_device_type(device_type: str) -> DeviceType | None:
    try:
        return DeviceType(device_type)
    except ValueError:
        return None


def get_employee(world: World, *, employee_id: str) -> Result:
    """Return the employee record, or an error if the id is unknown."""
    employee = world.employees.get(employee_id)
    if employee is None:
        return _err(f"employee not found: {employee_id}")
    return _ok(**employee.model_dump())


def get_policy(world: World, *, role: str) -> Result:
    """Return the access rules for a role, or an error for unknown roles."""
    policy = world.policies.get(role)
    if policy is None:
        return _err(f"policy not found: {role}")
    return _ok(role=role, access=policy.access_rules.get(role, []))


def get_inventory(world: World) -> Result:
    """Return current hardware inventory counts keyed by device type."""
    data = {device: world.inventory.get(device, 0) for device in DeviceType}
    return _ok(**data)


def assign_device(world: World, *, employee_id: str, device_type: str) -> Result:
    """Assign an available device; decrements inventory on success."""
    employee = world.employees.get(employee_id)
    if employee is None:
        return _err(f"employee not found: {employee_id}")

    device = _resolve_device_type(device_type)
    if device is None:
        return _err(f"unknown device type: {device_type}")

    count = world.inventory.get(device.value, 0)
    if count <= 0:
        return _err(f"inventory error: {device.value} not available")
    if employee.assigned_device is not None:
        return _err(f"employee already assigned: {employee.assigned_device.value}")

    world.inventory[device.value] = count - 1
    employee.assigned_device = device
    return _ok(employee_id=employee_id, device_type=device.value)


def create_procurement_request(
    world: World, *, employee_id: str, device_type: str
) -> Result:
    """Create an open procurement request; always succeeds when input is valid.

    The world deliberately does NOT check inventory here — the agent must
    check first. That ordering is the teachable failure mode.
    """
    employee = world.employees.get(employee_id)
    if employee is None:
        return _err(f"employee not found: {employee_id}")

    device = _resolve_device_type(device_type)
    if device is None:
        return _err(f"unknown device type: {device_type}")

    request_id = f"prq-{len(world.procurement) + 1}"
    world.procurement.append(
        ProcurementRequest(
            id=request_id, employee=employee_id, device_type=device, status="open"
        )
    )
    return _ok(request_id=request_id, status="open")


def grant_access(world: World, *, employee_id: str, access: str) -> Result:
    """Append an access grant to the employee.

    The world does NOT enforce that get_policy was called first — the
    deterministic grader checks that via scenario invariants.
    """
    employee = world.employees.get(employee_id)
    if employee is None:
        return _err(f"employee not found: {employee_id}")
    if not isinstance(access, str) or not access.strip():
        return _err("access must be a non-empty string")

    if access not in employee.granted_access:
        employee.granted_access.append(access)
    return _ok(employee_id=employee_id, access=access, granted=employee.granted_access)


def create_ticket(
    world: World, *, title: str, blocker: str | None = None, related_employee: str | None = None
) -> Result:
    """Create a ticket; a ticket with a blocker field set is an escalation."""
    if not title or not title.strip():
        return _err("title must be a non-empty string")
    if related_employee is not None and related_employee not in world.employees:
        return _err(f"employee not found: {related_employee}")

    ticket_id = f"tkt-{len(world.tickets) + 1}"
    world.tickets.append(
        Ticket(
            id=ticket_id,
            title=title,
            status="open" if blocker is None else "blocked",
            blocker=blocker,
            related_employee=related_employee,
        )
    )
    return _ok(ticket_id=ticket_id, status=world.tickets[-1].status)


def complete_onboarding(world: World, *, employee_id: str) -> Result:
    """Mark an employee completed, refusing while access or device are missing."""
    employee = world.employees.get(employee_id)
    if employee is None:
        return _err(f"employee not found: {employee_id}")

    required = world.policies.get(employee.role, None)
    if required is not None:
        missing = [
            grant
            for grant in required.access_rules.get(employee.role, [])
            if grant not in employee.granted_access
        ]
        if missing:
            return _err("cannot complete: access tasks blocked")
    if employee.assigned_device is None:
        return _err("cannot complete: no device assigned")

    employee.status = "completed"
    return _ok(employee_id=employee_id, status=employee.status)


def _parameters(
    required: list[str], properties: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": TOOL_GET_EMPLOYEE,
        "description": "Look up an employee record by id.",
        "parameters": _parameters(
            ["employee_id"], {"employee_id": {"type": "string"}}
        ),
    },
    {
        "name": TOOL_GET_POLICY,
        "description": "Get the required access grants for a role.",
        "parameters": _parameters(["role"], {"role": {"type": "string"}}),
    },
    {
        "name": TOOL_GET_INVENTORY,
        "description": "Get current hardware inventory counts (macbook, windows).",
        "parameters": _parameters([], {}),
    },
    {
        "name": TOOL_ASSIGN_DEVICE,
        "description": "Assign an available device to an employee.",
        "parameters": _parameters(
            ["employee_id", "device_type"],
            {
                "employee_id": {"type": "string"},
                "device_type": {"type": "string", "enum": ["macbook", "windows"]},
            },
        ),
    },
    {
        "name": TOOL_CREATE_PROCUREMENT_REQUEST,
        "description": "Create a procurement request for a device type.",
        "parameters": _parameters(
            ["employee_id", "device_type"],
            {
                "employee_id": {"type": "string"},
                "device_type": {"type": "string", "enum": ["macbook", "windows"]},
            },
        ),
    },
    {
        "name": TOOL_GRANT_ACCESS,
        "description": "Grant a single access entitlement to an employee.",
        "parameters": _parameters(
            ["employee_id", "access"],
            {
                "employee_id": {"type": "string"},
                "access": {"type": "string"},
            },
        ),
    },
    {
        "name": TOOL_CREATE_TICKET,
        "description": "Create a ticket; a blocker marks an escalation.",
        "parameters": _parameters(
            ["title"],
            {
                "title": {"type": "string"},
                "blocker": {"type": "string"},
                "related_employee": {"type": "string"},
            },
        ),
    },
    {
        "name": TOOL_COMPLETE_ONBOARDING,
        "description": "Mark onboarding complete once access and device are satisfied.",
        "parameters": _parameters(["employee_id"], {"employee_id": {"type": "string"}}),
    },
]

TOOL_SCHEMAS: list[dict[str, Any]] = list(_TOOL_SCHEMAS)


def tool_specs() -> list[dict[str, Any]]:
    """Schemas in the exact TOOL_SCHEMAS order (alias for the executor, M1)."""
    return [dict(schema) for schema in _TOOL_SCHEMAS]


_TOOLS: dict[str, Callable[..., Result]] = {
    TOOL_GET_EMPLOYEE: get_employee,
    TOOL_GET_POLICY: get_policy,
    TOOL_GET_INVENTORY: get_inventory,
    TOOL_ASSIGN_DEVICE: assign_device,
    TOOL_CREATE_PROCUREMENT_REQUEST: create_procurement_request,
    TOOL_GRANT_ACCESS: grant_access,
    TOOL_CREATE_TICKET: create_ticket,
    TOOL_COMPLETE_ONBOARDING: complete_onboarding,
}


def execute(world: World, tool: str, arguments: dict[str, Any] | None = None) -> Result:
    """Dispatch a named tool with kwargs; unknown tools return an error result.

    Signature is (world, tool_name, arguments) so M1's executor loop can call
    it uniformly; tools themselves keep the documented
    ``tool(world, *, kwarg=...)`` signature.
    """
    func = _TOOLS.get(tool)
    if func is None:
        return _err(f"unknown tool: {tool}")
    try:
        return func(world, **(arguments or {}))
    except TypeError as exc:  # missing/unknown keyword arguments
        return _err(f"invalid arguments for {tool}: {exc}")
