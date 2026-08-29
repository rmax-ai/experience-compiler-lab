"""Scenario dataset tests (M0)."""

import subprocess
import sys
from pathlib import Path

import pytest

from traces.schema import Scenario, load_scenarios
from world.api import TOOL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "datasets"
SPLIT_FILES = {
    "train": DATASETS / "train.jsonl",
    "validation": DATASETS / "validation.jsonl",
    "test": DATASETS / "test.jsonl",
}
SPLIT_SIZES = {"train": 30, "validation": 10, "test": 10}


@pytest.fixture(scope="module")
def all_scenarios() -> dict[str, list[Scenario]]:
    return {name: load_scenarios(str(path)) for name, path in SPLIT_FILES.items()}


def test_all_scenarios_load_and_validate() -> None:
    for name, path in SPLIT_FILES.items():
        assert path.exists(), f"{name} dataset missing: {path}"
        scenarios = load_scenarios(str(path))
        assert len(scenarios) > 0
        for scenario in scenarios:
            assert scenario.version == 1
            assert 1 <= scenario.difficulty <= 3


def test_split_sizes_and_uniqueness(all_scenarios: dict[str, list[Scenario]]) -> None:
    for name, expected in SPLIT_SIZES.items():
        assert len(all_scenarios[name]) == expected

    all_ids = [s.task_id for scenarios in all_scenarios.values() for s in scenarios]
    assert len(all_ids) == len(set(all_ids)), "task_id collisions across splits"
    assert len(all_ids) == 50

    all_seeds = [s.seed for scenarios in all_scenarios.values() for s in scenarios]
    assert len(all_seeds) == len(set(all_seeds)), "seed collisions across scenarios"


def test_regeneration_is_byte_identical() -> None:
    """Two fresh generation runs with the same seed produce byte-identical JSONL."""
    gen = REPO_ROOT / "src" / "world" / "fixtures" / "generate.py"

    def run() -> dict[str, bytes]:
        subprocess.run(
            [sys.executable, str(gen), "--seed", "42"],
            check=True,
            cwd=REPO_ROOT,
        )
        return {
            name: (DATASETS / f"{name}.jsonl").read_bytes()
            for name in SPLIT_FILES
        }

    first = run()
    second = run()
    for name in SPLIT_FILES:
        assert first[name] == second[name]


def test_toolset_is_subset_of_eight_tools(all_scenarios: dict[str, list[Scenario]]) -> None:
    for scenarios in all_scenarios.values():
        for scenario in scenarios:
            assert set(scenario.toolset) <= set(TOOL_NAMES)


def test_macbook_shortage_families_present(all_scenarios: dict[str, list[Scenario]]) -> None:
    """At least 3 scenarios carry the macbook:0 teachable shortage."""
    shortages = [
        s
        for scenarios in all_scenarios.values()
        for s in scenarios
        if s.world.inventory.get("macbook", 0) == 0
    ]
    assert len(shortages) >= 3
    for scenario in shortages:
        assert scenario.difficulty >= 2


def test_complete_onboarding_refusal_scenarios(all_scenarios: dict[str, list[Scenario]]) -> None:
    """At least 3 scenarios forbid marking onboarding complete."""
    refusals = [
        s
        for scenarios in all_scenarios.values()
        for s in scenarios
        if any(
            inv.path.startswith("employee.") and inv.path.endswith(".status")
            and inv.op == "=="
            and inv.value == "completed"
            for inv in s.grader.must_not
        )
    ]
    assert len(refusals) >= 3
    for scenario in refusals:
        assert scenario.grader.success_invariants  # positive outcomes exist too
