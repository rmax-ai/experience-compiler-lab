"""CLI mine smoke test: scripted runs -> knowledge records in a tmp store."""

import json
from pathlib import Path

from typer.testing import CliRunner

import cli
from agent.adapter import FakeModel
from experiments.runner import run_tasks
from knowledge.store import KnowledgeStore
from traces.schema import Scenario, load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

_USAGE = {"input_tokens": 100, "output_tokens": 40}


def _load_task(task_id: str) -> Scenario:
    scenarios = load_scenarios(str(REPO_ROOT / "datasets" / "validation.jsonl"))
    for scenario in scenarios:
        if scenario.task_id == task_id:
            return scenario
    raise AssertionError(f"scenario not found: {task_id}")


def _tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, sort_keys=True),
                },
            }
        ],
    }


def _answer(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _write_scripts(path: Path, entries: list[tuple[Scenario, list[dict]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for scenario, responses in entries:
            for response in responses:
                handle.write(
                    json.dumps(
                        {
                            "messages": [{"role": "user", "content": scenario.description}],
                            "response": response,
                            "usage": _USAGE,
                        }
                    )
                    + "\n"
                )


def _run_two_tasks(tmp_path: Path) -> TraceStore:
    """Run one scripted success and one scripted inventory failure."""
    success = _load_task("onboard_alice_basic_2")
    failure = _load_task("shortage_alice_macbook_3")

    success_scripts = [
        _tool_call("get_employee", {"employee_id": "alice"}, "c1"),
        _tool_call("get_policy", {"role": "engineer"}, "c2"),
        _tool_call("get_inventory", {}, "c3"),
        _tool_call("assign_device", {"employee_id": "alice", "device_type": "macbook"}, "c4"),
        _tool_call("grant_access", {"employee_id": "alice", "access": "vpn"}, "c5"),
        _tool_call("grant_access", {"employee_id": "alice", "access": "github"}, "c6"),
        _tool_call("complete_onboarding", {"employee_id": "alice"}, "c7"),
        _answer("Onboarding complete for Alice."),
    ]
    failure_scripts = [
        _tool_call("get_employee", {"employee_id": "alice"}, "c1"),
        _tool_call("get_inventory", {}, "c2"),
        _tool_call("assign_device", {"employee_id": "alice", "device_type": "macbook"}, "c3"),
        _tool_call("assign_device", {"employee_id": "alice", "device_type": "windows"}, "c4"),
        _tool_call("grant_access", {"employee_id": "alice", "access": "vpn"}, "c5"),
        _tool_call("grant_access", {"employee_id": "alice", "access": "github"}, "c6"),
        _answer("Device assigned; procurement and escalation pending."),
    ]

    scripts_path = tmp_path / "scripts.jsonl"
    _write_scripts(scripts_path, [(success, success_scripts), (failure, failure_scripts)])

    def factory() -> FakeModel:
        return FakeModel(scripts=scripts_path, model="fake", temperature=0.0)

    store = TraceStore(tmp_path)
    traces = run_tasks(
        scenarios=[success, failure],
        workflow="onboarding",
        model_factory=factory,
        seed=42,
        experiment_id="test-mine",
        store=store,
    )
    assert traces[0].outcome.success is True
    assert traces[1].outcome.success is False
    assert [t.run_id for t in traces] == ["run_001", "run_002"]
    return store


def test_cli_mine_produces_records(tmp_path: Path, monkeypatch) -> None:
    store = _run_two_tasks(tmp_path)
    knowledge = KnowledgeStore(tmp_path)

    monkeypatch.setattr(cli, "TraceStore", lambda: store)
    monkeypatch.setattr(cli, "KnowledgeStore", lambda: knowledge)

    result = runner.invoke(cli.app, ["mine", "--model", "fake"])

    assert result.exit_code == 0, result.output
    assert "records: 2" in result.output
    assert (
        "upserted check-available-inventory-before-assigning-hardware "
        "support=1 failures=1 confidence=0.50"
    ) in result.output
    assert (
        "upserted look-up-the-employee-record-before-granting-access "
        "support=1 failures=1 confidence=0.50"
    ) in result.output

    records = knowledge.all_records()
    assert len(records) == 2
    assert [record.id for record in records] == [
        "check-available-inventory-before-assigning-hardware",
        "look-up-the-employee-record-before-granting-access",
    ]
