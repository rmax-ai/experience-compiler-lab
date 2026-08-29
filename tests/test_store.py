"""TraceStore tests: append/get, immutability, index rebuild, run id allocation.

Also covers KnowledgeStore (knowledge/patterns + index.yaml): append-only
upsert merging, supersession, and byte-deterministic index regeneration.
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from knowledge.schema import Claim, ClaimType, Evidence, KnowledgeRecord, Scope, Statistics
from knowledge.store import KnowledgeStore
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


# -- KnowledgeStore -----------------------------------------------------------


def _record(
    record_id: str,
    supporting: list[str] | None = None,
    counterexamples: list[str] | None = None,
    *,
    first_seen: date = date(2026, 8, 1),
    last_updated: date = date(2026, 8, 1),
    **overrides: object,
) -> KnowledgeRecord:
    supporting = supporting or []
    counterexamples = counterexamples or []
    base = KnowledgeRecord(
        id=record_id,
        claim=Claim(type=ClaimType.warning, text=f"claim for {record_id}"),
        scope=Scope(workflows=["onboarding"]),
        evidence=Evidence(supporting_runs=supporting, counterexamples=counterexamples),
        statistics=Statistics(support=len(supporting), failures=len(counterexamples)),
        confidence=(len(supporting) + 1) / (len(supporting) + len(counterexamples) + 2),
        status="active",
        first_seen=first_seen,
        last_updated=last_updated,
    )
    return base.model_copy(update=overrides)


def test_upsert_new_writes_file_and_roundtrips(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    record = _record("check-inventory-before-assignment", supporting=["run_001"])
    returned = store.upsert(record)
    assert returned == record
    assert (tmp_path / "knowledge" / "patterns" / "check-inventory-before-assignment.yaml").exists()
    assert store.get("check-inventory-before-assignment") == record
    assert store.all_records() == [record]


def test_upsert_same_id_merges_evidence_and_keeps_first_seen(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    first = _record(
        "check-inventory-before-assignment",
        supporting=["run_001"],
        first_seen=date(2026, 8, 1),
        last_updated=date(2026, 8, 1),
    )
    store.upsert(first)

    second = _record(
        "check-inventory-before-assignment",
        supporting=["run_002", "run_001"],
        counterexamples=["run_010"],
        first_seen=date(2026, 8, 2),
        last_updated=date(2026, 8, 2),
    )
    merged = store.upsert(second)

    assert merged.evidence.supporting_runs == ["run_001", "run_002"]
    assert merged.evidence.counterexamples == ["run_010"]
    assert merged.statistics.support == 2
    assert merged.statistics.failures == 1
    assert merged.first_seen == date(2026, 8, 1)
    assert merged.last_updated == date.today()
    # confidence recomputed from the merged evidence: (2+1)/(2+1+2)
    assert merged.confidence == 0.6
    assert store.get("check-inventory-before-assignment") == merged


def test_upsert_never_removes_run_ids(tmp_path: Path) -> None:
    """Append-only guard: re-upserting with a smaller set must not shrink."""
    store = KnowledgeStore(tmp_path)
    store.upsert(_record("pattern", supporting=["run_001", "run_002"], counterexamples=["run_010"]))
    store.upsert(_record("pattern", supporting=["run_002"], counterexamples=[]))
    record = store.get("pattern")
    assert record.evidence.supporting_runs == ["run_001", "run_002"]
    assert record.evidence.counterexamples == ["run_010"]
    assert record.statistics.support == 2
    assert record.statistics.failures == 1


def test_supersede_marks_old_and_links_new(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    old = _record("old-claim", supporting=["run_001"])
    new = _record("new-claim", supporting=["run_002"])
    store.upsert(old)
    store.upsert(new)

    store.supersede("old-claim", "new-claim")

    superseded = store.get("old-claim")
    assert superseded.status == "superseded"
    assert superseded.superseded_by == ["new-claim"]
    assert store.get("new-claim").supersedes == ["old-claim"]
    # Nothing is deleted: both files still exist.
    assert {r.id for r in store.all_records()} == {"old-claim", "new-claim"}


def test_regenerate_index_byte_identical(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    store.upsert(_record("beta-claim", supporting=["run_001"], counterexamples=["run_010"]))
    store.upsert(_record("alpha-claim", supporting=["run_002"]))
    store.regenerate_index()
    first = (tmp_path / "knowledge" / "index.yaml").read_bytes()
    store.regenerate_index()
    second = (tmp_path / "knowledge" / "index.yaml").read_bytes()
    assert first == second

    index = (tmp_path / "knowledge" / "index.yaml").read_text(encoding="utf-8")
    assert "format_version: 1" in index
    assert "record_count: 2" in index
    # Records sorted by id; the file is the only index content (no timestamps).
    assert index.index("id: alpha-claim") < index.index("id: beta-claim")


def test_supersede_regenerates_index(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path)
    store.upsert(_record("old-claim"))
    store.upsert(_record("new-claim"))
    store.supersede("old-claim", "new-claim")
    index = (tmp_path / "knowledge" / "index.yaml").read_text(encoding="utf-8")
    assert "status: superseded" in index

