"""Deterministic scenario generator (docs/data-formats.md §1).

Generates the 50 M0 scenarios (30 train / 10 validation / 10 test) covering
the six teachable scenario families:

a. clean onboarding (success path)
b. inventory shortage (macbook count 0) — check before assigning; escalate
   via procurement request + blocker ticket
c. policy-required access — grants must match the role policy
d. missing information — unknown employee / missing policy must be escalated,
   not hallucinated
e. partial access — onboarding must NOT complete while access is incomplete;
   a blocker ticket is the correct escalation
f. update-the-ticket-after-checking-inventory — the ticket created before the
   inventory check must be updated afterwards

Idempotence: ``random.Random`` is re-seeded with the scenario seed and the
generator logic is deterministic, so the same seed + generator version always
produces byte-identical JSONL. No wall-clock timestamps anywhere.

Usage: ``python -m world.fixtures.generate [--seed N]`` (default seed 42).
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from traces.schema import Scenario, load_scenarios

# Synthetic enterprise data only (AGENTS.md §9).
EMPLOYEES: list[dict[str, str]] = [
    {"id": "alice", "name": "Alice", "role": "engineer", "department": "engineering"},
    {"id": "bob", "name": "Bob", "role": "designer", "department": "design"},
    {"id": "carol", "name": "Carol", "role": "engineer", "department": "engineering"},
    {"id": "dave", "name": "Dave", "role": "finance", "department": "finance"},
    {"id": "erin", "name": "Erin", "role": "designer", "department": "design"},
]

POLICIES: dict[str, dict[str, list[str]]] = {
    "engineer": {"access_rules": {"engineer": ["vpn", "github"]}},
    "designer": {"access_rules": {"designer": ["vpn", "figma"]}},
    "finance": {"access_rules": {"finance": ["vpn", "erp"]}},
}

TOOLSET: list[str] = [
    "get_employee",
    "get_policy",
    "get_inventory",
    "assign_device",
    "create_procurement_request",
    "grant_access",
    "create_ticket",
    "complete_onboarding",
]

TRAIN_COUNT = 30
VALIDATION_COUNT = 10
TEST_COUNT = 10
SPLITS: list[tuple[str, int]] = [
    ("train", TRAIN_COUNT),
    ("validation", VALIDATION_COUNT),
    ("test", TEST_COUNT),
]

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "datasets"


def _inv(path: str, op: str, value: Any) -> dict[str, Any]:
    return {"path": path, "op": op, "value": value}


def _world_spec(
    inventory: dict[str, int],
    employees: list[dict[str, Any]],
    *,
    policies: bool = True,
) -> dict[str, Any]:
    """Build the scenario ``world`` block for the given employees."""
    return {
        "inventory": dict(inventory),
        "employees": {emp["id"]: dict(emp) for emp in employees},
        "policies": (
            {role: dict(rules) for role, rules in POLICIES.items()}
            if policies
            else {}
        ),
        "documents": {
            "onboarding": {"id": "onboarding", "title": "Onboarding", "content": "text"}
        },
        "workflows": {"onboarding": ["assign_device", "grant_access", "verify", "close"]},
    }


def _employee_world(employee_id: str) -> list[dict[str, str]]:
    return [dict(e) for e in EMPLOYEES if e["id"] == employee_id]


def _suf(task_id: str, index: int) -> str:
    """Task ids are unique per instance: ``onboard_alice_basic``, ``..._1``, ..."""
    return task_id if index == 0 else f"{task_id}_{index}"


def _clean(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """a. Clean onboarding: success path."""
    employee = _employee_world("alice")[0]
    return {
        "task_id": _suf(f"onboard_{employee['id']}_basic", index),
        "description": (
            f"Onboard employee {employee['name']} ({employee['role']}). "
            "Fetch the policy, check inventory, assign a device, grant the "
            "required access, then complete onboarding."
        ),
        "world": _world_spec({"macbook": 3, "windows": 5}, [employee]),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv(f"employee.{employee['id']}.status", "==", "completed"),
                _inv(f"employee.{employee['id']}.assigned_device", "!=", None),
            ],
            "constraint_invariants": [
                _inv(f"employee.{employee['id']}.granted_access", "in", "vpn"),
                _inv(f"employee.{employee['id']}.granted_access", "in", "github"),
            ],
            "must_not": [],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _shortage(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """b. Inventory shortage: macbook count 0 (docs/data-formats.md §1 example).

    Mirrors the contract example: the correct agent checks inventory before
    assigning, assigns the available windows device, requests the missing
    macbook via procurement, and escalates the shortage with a blocker ticket.
    Assigning a macbook when the count is 0 is the trap.
    """
    employee = _employee_world("alice")[0]
    return {
        "task_id": _suf(f"shortage_{employee['id']}_macbook", index),
        "description": (
            f"Onboard employee {employee['name']} ({employee['role']}). Check "
            "the hardware inventory before assigning: macbooks are out of "
            "stock. Assign the available windows device, create a procurement "
            "request for the missing macbook, and escalate the shortage with a "
            "blocker ticket."
        ),
        "world": _world_spec({"macbook": 0, "windows": 5}, [employee]),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv(f"employee.{employee['id']}.status", "==", "completed"),
            ],
            "constraint_invariants": [
                _inv(
                    f"procurement.exists(employee={employee['id']})",
                    "==",
                    True,
                ),
                _inv("ticket.has_blocker(hardware unavailable)", "==", True),
            ],
            "must_not": [
                _inv(f"employee.{employee['id']}.assigned_device", "==", "macbook"),
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _shortage_no_inventory(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """b2. Inventory shortage with NO devices at all — must not complete."""
    employee = _employee_world("bob")[0]
    return {
        "task_id": _suf(f"shortage_{employee['id']}_none", index),
        "description": (
            f"Onboard employee {employee['name']}. No hardware is available at "
            "all; check inventory, create a procurement request, escalate the "
            "shortage with a blocker ticket, and do NOT mark onboarding "
            "complete."
        ),
        "world": _world_spec({"macbook": 0, "windows": 0}, [employee]),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv(
                    f"procurement.exists(employee={employee['id']})",
                    "==",
                    True,
                ),
                _inv("ticket.has_blocker(no hardware available)", "==", True),
            ],
            "constraint_invariants": [],
            "must_not": [
                _inv(f"employee.{employee['id']}.status", "==", "completed"),
                _inv(f"employee.{employee['id']}.assigned_device", "==", "macbook"),
                _inv(f"employee.{employee['id']}.assigned_device", "==", "windows"),
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _policy_access(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """c. Policy-required access: grants must match the role policy."""
    employee = _employee_world("carol")[0]
    return {
        "task_id": _suf(f"access_{employee['id']}_policy", index),
        "description": (
            f"Onboard employee {employee['name']} ({employee['role']}). Look up "
            "the access policy for the role first, then grant exactly the "
            "required access before completing onboarding."
        ),
        "world": _world_spec({"macbook": 2, "windows": 2}, [employee]),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv(f"employee.{employee['id']}.status", "==", "completed"),
                _inv(f"employee.{employee['id']}.assigned_device", "!=", None),
            ],
            "constraint_invariants": [
                _inv(f"employee.{employee['id']}.granted_access", "in", "github"),
                _inv(f"employee.{employee['id']}.granted_access", "in", "vpn"),
            ],
            "must_not": [
                # irrelevant access is a security trap: grants must match policy
                _inv(f"employee.{employee['id']}.granted_access", "in", "erp")
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _missing_employee(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """d. Missing information: unknown employee must be escalated, not invented."""
    return {
        "task_id": _suf("missing_employee_ghost", index),
        "description": (
            "Onboard employee Ghost, id 'ghost'. The employee record does not "
            "exist; do not fabricate one or mark onboarding complete. Create a "
            "blocker ticket to escalate the missing information."
        ),
        # ghost is deliberately NOT in the world — get_employee must fail.
        "world": _world_spec({"macbook": 2, "windows": 2}, []),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv("ticket.has_blocker(employee not found)", "==", True),
            ],
            "constraint_invariants": [],
            "must_not": [
                _inv("employee.ghost.status", "==", "completed"),
                _inv("employee.ghost.assigned_device", "==", "macbook"),
                _inv("employee.ghost.assigned_device", "==", "windows"),
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _missing_policy(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """d2. Missing information: policy absent; escalate, never complete."""
    employee = _employee_world("dave")[0]
    return {
        "task_id": _suf(f"missing_policy_{employee['id']}", index),
        "description": (
            f"Onboard employee {employee['name']} ({employee['role']}). The "
            "access policy for the role is missing; without it the required "
            "access cannot be verified. Escalate with a blocker ticket instead "
            "of guessing access."
        ),
        "world": _world_spec({"macbook": 2, "windows": 2}, [employee], policies=False),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv("ticket.has_blocker(policy not found)", "==", True),
            ],
            "constraint_invariants": [],
            "must_not": [
                _inv(f"employee.{employee['id']}.status", "==", "completed")
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _partial_access(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """e. Partial access: some grants blocked; never mark onboarding complete."""
    employee = _employee_world("erin")[0]
    employee = dict(employee, granted_access=["vpn"])  # figma still pending
    return {
        "task_id": _suf(f"partial_access_{employee['id']}", index),
        "description": (
            f"Onboard employee {employee['name']} ({employee['role']}). The "
            "figma access request is stuck in approval and cannot be granted "
            "yet. Grant what you can, record the blocked access in a blocker "
            "ticket, and do NOT mark onboarding complete."
        ),
        "world": _world_spec({"macbook": 2, "windows": 2}, [employee]),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv("ticket.has_blocker(access blocked)", "==", True),
            ],
            "constraint_invariants": [
                _inv(f"employee.{employee['id']}.granted_access", "in", "vpn"),
            ],
            "must_not": [
                _inv(f"employee.{employee['id']}.status", "==", "completed")
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


def _update_ticket(seed: int, difficulty: int, index: int = 0) -> dict[str, Any]:
    """f. Update-the-ticket-after-checking-inventory style task."""
    employee = _employee_world("alice")[0]
    return {
        "task_id": _suf(f"update_ticket_{employee['id']}", index),
        "description": (
            f"Onboard employee {employee['name']}. A ticket about the macbook "
            "shortage was filed before the inventory check. Check the "
            "inventory first, then update the ticket to reflect the actual "
            "inventory (blocker should note the inventory was checked and "
            "macbooks are unavailable) and escalate the shortage."
        ),
        "world": _world_spec({"macbook": 0, "windows": 5}, [employee]),
        "toolset": list(TOOLSET),
        "grader": {
            "success_invariants": [
                _inv("ticket.has_blocker(inventory checked)", "==", True),
                _inv("ticket.has_blocker(macbook unavailable)", "==", True),
            ],
            "constraint_invariants": [],
            "must_not": [
                _inv(f"employee.{employee['id']}.status", "==", "completed"),
                _inv(f"employee.{employee['id']}.assigned_device", "==", "macbook"),
                # hallucinating inventory in the ticket is a trap
                _inv("ticket.has_blocker(macbook available)", "==", True),
            ],
        },
        "seed": seed,
        "difficulty": difficulty,
        "version": 1,
    }


# (name, factory, difficulty, count) — deterministic generation order.
Factory = Callable[[int, int, int], dict[str, Any]]
FAMILIES: list[tuple[str, Factory, int, int]] = [
    ("clean", _clean, 1, 10),
    ("shortage", _shortage, 2, 9),
    ("shortage_no_inventory", _shortage_no_inventory, 2, 5),
    ("policy_access", _policy_access, 2, 10),
    ("missing_employee", _missing_employee, 3, 3),
    ("missing_policy", _missing_policy, 3, 3),
    ("partial_access", _partial_access, 3, 6),
    ("update_ticket", _update_ticket, 3, 4),
]
assert sum(count for _, _, _, count in FAMILIES) == 50


def _sliced(
    factories: list[tuple[str, Factory, int, int]], seed: int
) -> list[dict[str, Any]]:
    """Slice each family's scenarios into the requested counts (deterministic).

    Each instance gets a unique task_id (via its per-family index) and a
    unique seed derived from the base seed, so scenario ids and seeds never
    collide across the 50 scenarios.
    """
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []
    counter = 0
    for _, factory, difficulty, count in factories:
        for index in range(count):
            scenario = factory(seed, difficulty, index)
            scenario["seed"] = seed + counter
            counter += 1
            scenarios.append(scenario)
    rng.shuffle(scenarios)  # deterministic given `seed`
    return scenarios


def build(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Generate all three splits as a dict of scenario dicts."""
    all_scenarios = _sliced(FAMILIES, seed)
    return {
        "train": all_scenarios[:TRAIN_COUNT],
        "validation": all_scenarios[TRAIN_COUNT : TRAIN_COUNT + VALIDATION_COUNT],
        "test": all_scenarios[TRAIN_COUNT + VALIDATION_COUNT :],
    }


def _validate_scenarios(path: Path) -> None:
    loaded = load_scenarios(str(path))
    if len(loaded) == 0:
        raise RuntimeError(f"no scenarios loaded from {path}")
    # self-check every grader spec against the api toolset
    for scenario in loaded:
        unknown = set(scenario.toolset) - set(TOOLSET)
        if unknown:
            raise RuntimeError(
                f"{scenario.task_id}: unknown tools {sorted(unknown)}"
            )


def write(seed: int = 42) -> list[Path]:
    """Write the three JSONL splits and validate them; returns written paths."""
    splits = build(seed)
    written: list[Path] = []
    for name, count in SPLITS:
        path = OUTPUT_DIR / f"{name}.jsonl"
        scenarios = splits[name]
        if len(scenarios) != count:
            raise RuntimeError(f"{name}: expected {count} scenarios, got {len(scenarios)}")
        with path.open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                handle.write(Scenario.model_validate(scenario).model_dump_json() + "\n")
        _validate_scenarios(path)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic M0 scenario datasets."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="generator seed (default: 42)"
    )
    args = parser.parse_args()

    paths = write(args.seed)
    for path in paths:
        print(f"wrote {path} ({sum(1 for _ in path.open(encoding='utf-8'))} scenarios)")


if __name__ == "__main__":
    main()
