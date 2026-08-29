"""Deterministic grader tests (M0)."""

from evals.graders import evaluate
from traces.schema import GraderSpec, Invariant
from world.state import Employee, Policy, World

ENGINEER_POLICY = Policy(access_rules={"engineer": ["vpn", "github"]})


def _world(
    *,
    status: str = "pending",
    assigned_device: str | None = None,
    granted: list[str] | None = None,
    inventory: dict[str, int] | None = None,
) -> World:
    return World(
        inventory={"macbook": 3, "windows": 5} if inventory is None else inventory,
        employees={
            "alice": Employee(
                id="alice",
                name="Alice",
                role="engineer",
                department="engineering",
                status=status,
                assigned_device=assigned_device,
                granted_access=list(granted or []),
            )
        },
        policies={"engineer": ENGINEER_POLICY},
        documents={
            "onboarding": {
                "id": "onboarding",
                "title": "Onboarding",
                "content": "text",
            }
        },
        workflows={"onboarding": ["assign_device", "grant_access", "verify", "close"]},
    )


def _spec(
    success: list[Invariant] | None = None,
    constraints: list[Invariant] | None = None,
    must_not: list[Invariant] | None = None,
) -> GraderSpec:
    return GraderSpec(
        success_invariants=success or [],
        constraint_invariants=constraints or [],
        must_not=must_not or [],
    )


def test_success_invariants_alone_determine_success() -> None:
    world = _world(status="completed", assigned_device="macbook")
    spec = _spec(success=[Invariant(path="employee.alice.status", op="==", value="completed")])
    result = evaluate(world, spec)
    assert result.success is True
    assert result.errors == []
    assert result.violated_constraints == []

    failing = _spec(
        success=[Invariant(path="employee.alice.status", op="==", value="pending")]
    )
    bad = evaluate(world, failing)
    assert bad.success is False
    assert len(bad.errors) == 1
    assert bad.violated_constraints == []


def test_violated_constraints_do_not_flip_success() -> None:
    world = _world(status="completed", assigned_device="macbook")
    spec = _spec(
        success=[Invariant(path="employee.alice.status", op="==", value="completed")],
        constraints=[
            Invariant(
                path="procurement.exists(employee=alice)", op="==", value=True
            ),
            Invariant(
                path="ticket.has_blocker(hardware unavailable)", op="==", value=True
            ),
        ],
    )
    result = evaluate(world, spec)
    assert result.success is True
    assert result.errors == []
    assert sorted(result.violated_constraints) == [
        "invariant failed: procurement.exists(employee=alice) == True (actual: False)",
        "invariant failed: ticket.has_blocker(hardware unavailable) == True (actual: False)",
    ]


def test_must_not_flip_success_false_and_record_error() -> None:
    world = _world(status="completed", assigned_device="macbook")
    spec = _spec(
        success=[Invariant(path="employee.alice.status", op="==", value="completed")],
        must_not=[Invariant(path="employee.alice.assigned_device", op="==", value="macbook")],
    )
    result = evaluate(world, spec)
    assert result.success is False
    assert result.errors == [
        "must_not violated: employee.alice.assigned_device == macbook"
    ]
    assert result.violated_constraints == []


def test_all_four_ops() -> None:
    world = _world(
        status="completed",
        assigned_device="windows",
        granted=["vpn", "github"],
        inventory={"macbook": 0, "windows": 4},
    )
    spec = _spec(
        success=[
            Invariant(path="employee.alice.status", op="==", value="completed"),
            Invariant(path="employee.alice.assigned_device", op="!=", value="macbook"),
            Invariant(path="employee.alice.granted_access", op="in", value="github"),
            Invariant(path="procurement.exists(employee=alice)", op="exists"),
        ],
    )
    # procurement request present -> exists passes
    from world.state import ProcurementRequest

    world.procurement.append(
        ProcurementRequest(id="prq-1", employee="alice", device_type="macbook")
    )
    result = evaluate(world, spec)
    assert result.success is True

    # remove the procurement request -> exists fails
    world.procurement.clear()
    result2 = evaluate(world, spec)
    assert result2.success is False
    assert any("procurement.exists(employee=alice) exists" in e for e in result2.errors)


def test_deterministic_violation_ordering() -> None:
    world = _world(status="pending", assigned_device=None, granted=[])
    spec = _spec(
        constraints=[
            Invariant(path="employee.alice.granted_access", op="in", value="github"),
            Invariant(path="employee.alice.granted_access", op="in", value="vpn"),
            Invariant(path="employee.alice.status", op="==", value="pending"),
        ]
    )
    first = evaluate(world, spec)
    second = evaluate(world, spec)
    assert first == second
    # both failing "in" constraints reported, sorted; the == constraint holds
    assert first.violated_constraints == [
        "invariant failed: employee.alice.granted_access in 'github' (actual: [])",
        "invariant failed: employee.alice.granted_access in 'vpn' (actual: [])",
    ]


def test_missing_employee_path_resolves_none() -> None:
    """employee.<id> paths for unknown employees resolve to None, so
    ``must_not`` can never fire and equality invariants fail cleanly."""
    world = _world()
    spec = _spec(
        must_not=[Invariant(path="employee.ghost.status", op="==", value="completed")],
        success=[Invariant(path="employee.ghost.status", op="==", value="pending")],
    )
    result = evaluate(world, spec)
    assert result.success is False  # ghost is not pending — it does not exist
    assert result.errors == [
        "invariant failed: employee.ghost.status == 'pending' (actual: None)"
    ]
