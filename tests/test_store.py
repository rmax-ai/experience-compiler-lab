"""TraceStore tests: append/get, immutability, index rebuild, run id allocation."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from traces.schema import (
    Action,
    Manifest,
    Message,
    Metrics,
    Outcome,
    Trace,
)
from traces.store import TraceStore


def _make_trace(run_id: str, task_id: str = "onboard_alice_basic") -> Trace:
    timestamp = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)  # noqa: UP017
    return Trace(
        run_id=run_id,
        task_id=task_id,
        model="fake",
        skill_version="1",
        messages=[
            Message(role="system", content="You are an agent."),
            Message(role="user", content=f"Task {task_id}."),
            Message(role="assistant", content="Done."),
        ],
        actions=[
            Action(
                index=0,
                tool="get_employee",
                arguments={"employee_id": "alice"},
                result={"ok": True, "error": None, "data": {"id": "alice"}},
                timestamp=timestamp,
                world_state_before="inventory:macbook=3",
                world_state_after="inventory:macbook=3",
            )
        ],
        final_answer="Done.",
        outcome=Outcome(success=True, errors=[], violated_constraints=[]),
        metrics=Metrics(
            tool_calls=1,
            tokens_in=100,
            tokens_out=50,
            estimated_cost_usd=0.00009,
            latency_s=0.5,
            recovery_count=0,
            trajectory_length=3,
        ),
        manifest=Manifest(
            experiment_id="test",
            model={"name": "fake", "temperature": 0.0},
            dataset_version="a",
            skill_version="b",
            knowledge_version="c",
            environment_version="d",
            seed=1,
        ),
    )


def test_append_then_get_roundtrip(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = _make_trace("run_001")
    store.append(trace)
    restored = store.get("run_001")
    assert restored.model_dump() == trace.model_dump()
    assert (tmp_path / "experience" / "runs" / "run_001.jsonl").exists()


def test_append_same_run_id_twice_raises(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = _make_trace("run_001")
    store.append(trace)
    with pytest.raises(FileExistsError):
        store.append(trace)


def test_rebuild_index_restores_identical_rows(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_make_trace("run_001", task_id="task_a"))
    store.append(_make_trace("run_002", task_id="task_b"))

    before = store.list_runs()
    assert [row["run_id"] for row in before] == ["run_001", "run_002"]

    store.rebuild_index()
    after = store.list_runs()
    assert after == before


def test_rebuild_index_recovers_from_missing_table(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_make_trace("run_001"))
    store.rebuild_index()
    assert [row["run_id"] for row in store.list_runs()] == ["run_001"]


def test_list_runs_filters(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_make_trace("run_001", task_id="task_a"))
    store.append(_make_trace("run_002", task_id="task_b"))
    assert [r["run_id"] for r in store.list_runs(task_id="task_a")] == ["run_001"]
    assert [r["run_id"] for r in store.list_runs(model="fake")] == ["run_001", "run_002"]
    assert [r["run_id"] for r in store.list_runs(success=True)] == ["run_001", "run_002"]
    assert store.list_runs(success=False) == []


def test_next_run_id_deterministic_on_empty_store(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    assert store.next_run_id() == "run_001"


def test_next_run_id_increments_after_appends(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_make_trace(store.next_run_id()))
    assert store.next_run_id() == "run_002"
    store.append(_make_trace(store.next_run_id()))
    assert store.next_run_id() == "run_003"
