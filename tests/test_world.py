"""Deterministic world + tool tests (M0)."""

from world.api import (
    assign_device,
    complete_onboarding,
    create_procurement_request,
    create_ticket,
    get_employee,
    get_inventory,
    get_policy,
    grant_access,
)
from world.state import DeviceType, Employee, Policy, World

ENGINEER_POLICY = Policy(access_rules={"engineer": ["vpn", "github"]})


def _world(inventory: dict[str, int] | None = None, employee_id: str = "alice") -> World:
    return World(
        inventory={"macbook": 3, "windows": 5} if inventory is None else inventory,
        employees={
            employee_id: Employee(
                id=employee_id,
                name="Alice",
                role="engineer",
                department="engineering",
            )
        },
        policies={"engineer": ENGINEER_POLICY},
        documents={
            "onboarding": {
                "id": "onboarding",
                "title": "Onboarding",
                "content": "onboard every employee",
            }
        },
        workflows={"onboarding": ["assign_device", "grant_access", "verify", "close"]},
    )


def test_snapshot_determinism() -> None:
    """Two identically-constructed worlds have byte-identical snapshots."""
    w1 = _world()
    w2 = _world()
    assert w1.snapshot() == w2.snapshot()
    assert type(w1.snapshot()) is str


def test_reset_restores_initial_state_exactly() -> None:
    """Mutate the world, reset it, and the snapshot matches the original."""
    world = _world()
    before = world.snapshot()

    assign_device(world, employee_id="alice", device_type="macbook")
    grant_access(world, employee_id="alice", access="vpn")
    grant_access(world, employee_id="alice", access="github")
    create_procurement_request(world, employee_id="alice", device_type="windows")
    create_ticket(world, title="blocked", blocker="something broke")
    world.reset()

    assert world.snapshot() == before
    assert world.inventory == {"macbook": 3, "windows": 5}
    assert world.employees["alice"].status == "pending"
    assert world.employees["alice"].assigned_device is None
    assert world.employees["alice"].granted_access == []
    assert world.tickets == []
    assert world.procurement == []


def test_reset_is_repeatable() -> None:
    """reset() can be called again after further mutation."""
    world = _world()
    baseline = world.snapshot()
    for _ in range(3):
        assign_device(world, employee_id="alice", device_type="macbook")
        world.reset()
        assert world.snapshot() == baseline


def test_assign_device_decrements_inventory_on_success() -> None:
    world = _world(inventory={"macbook": 3, "windows": 5})
    result = assign_device(world, employee_id="alice", device_type="macbook")
    assert result["ok"] is True
    assert result["error"] is None
    assert result["data"] is not None
    assert result["data"]["device_type"] == "macbook"
    assert world.inventory["macbook"] == 2
    assert world.employees["alice"].assigned_device == DeviceType.MACBOOK


def test_assign_device_fails_on_empty_inventory() -> None:
    world = _world(inventory={"macbook": 0, "windows": 5})
    result = assign_device(world, employee_id="alice", device_type="macbook")
    assert result["ok"] is False
    assert result["error"] is not None
    assert result["error"].startswith("inventory error")
    assert world.employees["alice"].assigned_device is None
    assert world.inventory["macbook"] == 0


def test_assign_device_fails_for_missing_employee() -> None:
    world = _world()
    result = assign_device(world, employee_id="nobody", device_type="macbook")
    assert result["ok"] is False
    assert result["error"] == "employee not found: nobody"


def test_assign_device_refuses_duplicate_assignment() -> None:
    world = _world()
    assign_device(world, employee_id="alice", device_type="macbook")
    second = assign_device(world, employee_id="alice", device_type="windows")
    assert second["ok"] is False
    assert world.employees["alice"].assigned_device == DeviceType.MACBOOK
    assert world.inventory["windows"] == 5  # untouched by the failed attempt


def test_complete_onboarding_refuses_without_device() -> None:
    world = _world()
    grant_access(world, employee_id="alice", access="vpn")
    grant_access(world, employee_id="alice", access="github")
    result = complete_onboarding(world, employee_id="alice")
    assert result["ok"] is False
    assert result["error"] == "cannot complete: no device assigned"
    assert world.employees["alice"].status == "pending"


def test_complete_onboarding_refuses_when_access_blocked() -> None:
    world = _world()
    assign_device(world, employee_id="alice", device_type="macbook")
    grant_access(world, employee_id="alice", access="vpn")  # github missing
    result = complete_onboarding(world, employee_id="alice")
    assert result["ok"] is False
    assert result["error"] == "cannot complete: access tasks blocked"
    assert world.employees["alice"].status == "pending"


def test_complete_onboarding_succeeds_with_device_and_access() -> None:
    world = _world()
    assign_device(world, employee_id="alice", device_type="macbook")
    grant_access(world, employee_id="alice", access="vpn")
    grant_access(world, employee_id="alice", access="github")
    result = complete_onboarding(world, employee_id="alice")
    assert result["ok"] is True
    assert result["error"] is None
    assert world.employees["alice"].status == "completed"


def test_create_procurement_request_always_succeeds() -> None:
    world = _world(inventory={"macbook": 3, "windows": 5})
    result = create_procurement_request(world, employee_id="alice", device_type="macbook")
    assert result["ok"] is True
    assert result["data"] is not None
    assert result["data"]["request_id"] == "prq-1"
    # succeeds even though the macbook inventory is nonzero
    assert world.inventory["macbook"] == 3
    assert len(world.procurement) == 1
    assert world.procurement[0].employee == "alice"
    assert world.procurement[0].device_type == DeviceType.MACBOOK


def test_create_ticket_with_blocker_records_escalation() -> None:
    world = _world()
    result = create_ticket(world, title="shortage", blocker="hardware unavailable")
    assert result["ok"] is True
    assert result["data"] is not None
    assert result["data"]["ticket_id"] == "tkt-1"
    assert len(world.tickets) == 1
    ticket = world.tickets[0]
    assert ticket.title == "shortage"
    assert ticket.blocker == "hardware unavailable"
    assert ticket.status == "blocked"
    assert ticket.related_employee is None


def test_get_employee_and_get_policy() -> None:
    world = _world()
    emp = get_employee(world, employee_id="alice")
    assert emp["ok"] is True
    assert emp["data"]["role"] == "engineer"

    missing = get_employee(world, employee_id="ghost")
    assert missing["ok"] is False
    assert missing["error"] == "employee not found: ghost"

    policy = get_policy(world, role="engineer")
    assert policy["ok"] is True
    assert policy["data"]["access"] == ["vpn", "github"]

    unknown = get_policy(world, role="cto")
    assert unknown["ok"] is False
    assert unknown["error"] == "policy not found: cto"


def test_get_inventory_reports_both_types() -> None:
    world = _world()
    result = get_inventory(world)
    assert result["ok"] is True
    assert result["data"] == {"macbook": 3, "windows": 5}


def test_grant_access_appends() -> None:
    world = _world()
    result = grant_access(world, employee_id="alice", access="vpn")
    assert result["ok"] is True
    assert world.employees["alice"].granted_access == ["vpn"]
    # granting the same access twice is idempotent
    grant_access(world, employee_id="alice", access="vpn")
    assert world.employees["alice"].granted_access == ["vpn"]
