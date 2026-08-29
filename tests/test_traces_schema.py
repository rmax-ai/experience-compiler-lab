"""Trace schema tests: JSONL roundtrip, dataset load, extra-field rejection."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from traces.schema import (
    Action,
    Manifest,
    Message,
    Metrics,
    Outcome,
    Trace,
    from_jsonl,
    git_short_hash,
    load_scenarios,
    to_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _full_trace() -> Trace:
    timestamp = datetime(2026, 8, 29, 12, 0, 0, 123456, tzinfo=timezone.utc)  # noqa: UP017
    return Trace(
        run_id="run_001",
        task_id="onboard_alice_basic",
        model="deepseek-v4-flash",
        skill_version="1",
        messages=[
            Message(role="system", content="You are an agent operating enterprise tools."),
            Message(role="user", content="Onboard employee Alice."),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_employee",
                            "arguments": '{"employee_id": "alice"}',
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                content='{"ok": true, "error": null, "data": {"id": "alice"}}',
            ),
            Message(role="assistant", content="Onboarding complete."),
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
        final_answer="Onboarding complete.",
        outcome=Outcome(success=True, errors=[], violated_constraints=[]),
        metrics=Metrics(
            tool_calls=1,
            tokens_in=100,
            tokens_out=50,
            estimated_cost_usd=0.00009,
            latency_s=1.2345,
            recovery_count=0,
            trajectory_length=5,
        ),
        manifest=Manifest(
            experiment_id="exp-001",
            model={"name": "deepseek-v4-flash", "temperature": 0.0},
            dataset_version="abc1234",
            skill_version="def5678",
            knowledge_version="0123456",
            environment_version="7890abc",
            seed=42,
        ),
    )


def test_trace_jsonl_roundtrip_is_lossless() -> None:
    trace = _full_trace()
    line = to_jsonl(trace)
    assert line.endswith("\n") is False, "no trailing newline in to_jsonl"
    restored = from_jsonl(line)
    assert restored.model_dump() == trace.model_dump()


def test_trace_jsonl_roundtrip_is_single_line() -> None:
    line = to_jsonl(_full_trace())
    assert "\n" not in line


def test_all_50_committed_scenarios_still_load() -> None:
    total = 0
    for split in ("train", "validation", "test"):
        scenarios = load_scenarios(str(REPO_ROOT / "datasets" / f"{split}.jsonl"))
        assert len(scenarios) > 0
        total += len(scenarios)
    assert total == 50


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        Metrics(
            tool_calls=1,
            tokens_in=1,
            tokens_out=1,
            estimated_cost_usd=0.0,
            latency_s=0.0,
            recovery_count=0,
            trajectory_length=1,
            bogus=1,
        )
    with pytest.raises(ValidationError):
        Outcome(success=True, errors=[], bogus=1)
    with pytest.raises(ValidationError):
        Action(
            index=0,
            tool="get_employee",
            arguments={},
            result={},
            timestamp=datetime.now(timezone.utc),  # noqa: UP017
            world_state_before="",
            world_state_after="",
            bogus=1,
        )
    with pytest.raises(ValidationError):
        Manifest(
            experiment_id="e",
            model={"name": "m", "temperature": 0.0},
            dataset_version="a",
            skill_version="b",
            knowledge_version="c",
            environment_version="d",
            seed=1,
            bogus=1,
        )


def test_git_short_hash_returns_string() -> None:
    version = git_short_hash()
    assert isinstance(version, str)
    assert len(version) > 0
