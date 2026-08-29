"""Miner tests: prompt/summary determinism, JSON parsing, evidence linking."""

import json
from datetime import datetime, timezone

from agent.adapter import FakeModel
from knowledge.miner import CandidateEvidence, merge, mine, summarize_trace
from traces.schema import Action, Manifest, Message, Metrics, Outcome, Trace

_TIMESTAMP = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)  # noqa: UP017


def _make_trace(run_id: str, actions: list[dict], success: bool) -> Trace:
    trace_actions = [
        Action(
            index=index,
            tool=spec["tool"],
            arguments={},
            result=spec["result"],
            timestamp=_TIMESTAMP,
            world_state_before="",
            world_state_after="",
        )
        for index, spec in enumerate(actions)
    ]
    return Trace(
        run_id=run_id,
        task_id="onboard_alice_basic",
        model="fake",
        skill_version="1",
        messages=[Message(role="user", content="Task.")],
        actions=trace_actions,
        final_answer="Done.",
        outcome=Outcome(success=success, errors=[], violated_constraints=[]),
        metrics=Metrics(
            tool_calls=len(trace_actions),
            tokens_in=100,
            tokens_out=50,
            estimated_cost_usd=0.00009,
            latency_s=0.5,
            recovery_count=0,
            trajectory_length=1,
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


def _ok(tool: str) -> dict:
    return {"tool": tool, "result": {"ok": True, "error": None}}


def _fail(tool: str, error: str) -> dict:
    return {"tool": tool, "result": {"ok": False, "error": error}}


def _script(content: str) -> tuple[dict, dict]:
    return (
        {"role": "assistant", "content": content},
        {"input_tokens": 50, "output_tokens": 30},
    )


# -- summarize_trace ---------------------------------------------------------


def test_summarize_trace_deterministic() -> None:
    actions = [_ok("get_inventory"), _fail("assign_device", "inventory error: macbook not available")]
    a = _make_trace("run_001", actions, success=False)
    b = _make_trace("run_001", actions, success=False)
    assert summarize_trace(a) == summarize_trace(b)


def test_summarize_trace_content() -> None:
    trace = _make_trace(
        "run_002",
        [_ok("get_inventory"), _fail("assign_device", "inventory error: macbook not available")],
        success=False,
    )
    summary = summarize_trace(trace)
    assert summary["run_id"] == "run_002"
    assert summary["success"] is False
    assert summary["tool_call_count"] == 2
    assert summary["tool_sequence"] == [
        {"tool": "get_inventory", "ok": True, "error_key": ""},
        {"tool": "assign_device", "ok": False, "error_key": "inventory error: macbook not available"},
    ]
    assert summary["errors"] == []


# -- mine() ------------------------------------------------------------------


def test_mine_parses_fixed_json_array() -> None:
    payload = [
        {
            "kind": "repeated_failure_mode",
            "hypothesis": "Check inventory first",
            "mentioned_tools": ["assign_device"],
            "mentioned_errors": ["inventory error"],
        }
    ]
    model = FakeModel(scripts=[_script(json.dumps(payload))])
    traces = [_make_trace("run_001", [_fail("assign_device", "inventory error")], success=False)]
    result = mine(traces, model)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.kind.value == "repeated_failure_mode"
    assert candidate.hypothesis == "Check inventory first"
    assert candidate.mentioned_tools == ["assign_device"]
    assert result.parse_failures == 0
    assert result.model_calls == 1
    assert result.tokens == 80
    assert result.cost_usd > 0.0


def test_mine_extracts_fenced_json_from_prose() -> None:
    payload = [
        {
            "kind": "repeated_success_strategy",
            "hypothesis": "Grant access after policy lookup",
            "mentioned_tools": ["get_policy", "grant_access"],
            "mentioned_errors": [],
        }
    ]
    prose = f"Here you go:\n```json\n{json.dumps(payload)}\n```\nHope that helps."
    model = FakeModel(scripts=[_script(prose)])
    result = mine([_make_trace("run_001", [_ok("get_policy")], success=True)], model)
    assert len(result.candidates) == 1
    assert result.candidates[0].hypothesis == "Grant access after policy lookup"
    assert result.parse_failures == 0


def test_mine_drops_invalid_items_and_counts_them() -> None:
    payload = [
        {"kind": "repeated_failure_mode", "hypothesis": "valid one", "mentioned_tools": [], "mentioned_errors": []},
        {"kind": "not_a_kind", "hypothesis": "invalid kind"},
        "garbage",
        {"kind": "missing_hypothesis", "mentioned_tools": []},
    ]
    model = FakeModel(scripts=[_script(json.dumps(payload))])
    result = mine([_make_trace("run_001", [_fail("assign_device", "inventory error")], success=False)], model)
    assert len(result.candidates) == 1
    assert result.parse_failures == 3
    assert result.model_calls == 1


def test_mine_retries_once_then_returns_empty_on_garbage_twice() -> None:
    model = FakeModel(
        scripts=[
            _script("this is not json"),
            _script("still not json {{{"),
        ]
    )
    result = mine([_make_trace("run_001", [_ok("get_inventory")], success=True)], model)
    assert result.candidates == []
    assert result.parse_failures >= 1
    assert result.model_calls == 2


def test_mine_repair_prompt_recovers_valid_json() -> None:
    model = FakeModel(
        scripts=[
            _script("oops I wrote prose"),
            _script(json.dumps([{"kind": "termination_failure", "hypothesis": "Stopped early"}])),
        ]
    )
    result = mine([_make_trace("run_001", [], success=False)], model)
    assert len(result.candidates) == 1
    assert result.candidates[0].hypothesis == "Stopped early"
    assert result.parse_failures == 1
    assert result.model_calls == 2


# -- merge() -----------------------------------------------------------------


def _evidence_traces() -> list[Trace]:
    failure = _make_trace(
        "run_002",
        [_ok("get_inventory"), _fail("assign_device", "inventory error: macbook not available")],
        success=False,
    )
    success = _make_trace(
        "run_001",
        [
            _ok("get_employee"),
            _ok("get_inventory"),
            _ok("assign_device"),
            _ok("grant_access"),
        ],
        success=True,
    )
    return [success, failure]


def test_merge_deterministic() -> None:
    candidates = [
        CandidateEvidence(
            kind="repeated_failure_mode",
            hypothesis="Check inventory before assigning",
            mentioned_tools=["assign_device"],
            mentioned_errors=["inventory error"],
        ),
        CandidateEvidence(
            kind="repeated_success_strategy",
            hypothesis="Look up employee before granting access",
            mentioned_tools=["get_employee", "grant_access"],
            mentioned_errors=[],
        ),
    ]
    traces = _evidence_traces()
    first = merge(candidates, traces)
    second = merge(candidates, traces)
    assert [record.model_dump() for record in first] == [record.model_dump() for record in second]
    assert [record.id for record in first] == [record.id for record in second]


def test_merge_evidence_linking_failure_kind() -> None:
    candidates = [
        CandidateEvidence(
            kind="repeated_failure_mode",
            hypothesis="Check inventory before assigning",
            mentioned_tools=["assign_device"],
            mentioned_errors=["inventory error"],
        )
    ]
    records = merge(candidates, _evidence_traces())
    assert len(records) == 1
    record = records[0]
    assert record.evidence.supporting_runs == ["run_002"]
    assert record.evidence.counterexamples == ["run_001"]
    assert record.statistics.support == 1
    assert record.statistics.failures == 1
    assert record.confidence == 0.5
    assert record.claim.type.value == "warning"
    assert record.claim.text == "Check inventory before assigning"
    assert record.scope.workflows == ["onboarding"]


def test_merge_evidence_linking_success_kind() -> None:
    candidates = [
        CandidateEvidence(
            kind="repeated_success_strategy",
            hypothesis="Look up employee before granting access",
            mentioned_tools=["get_employee", "grant_access"],
            mentioned_errors=[],
        )
    ]
    records = merge(candidates, _evidence_traces())
    assert len(records) == 1
    record = records[0]
    assert record.evidence.supporting_runs == ["run_001"]
    assert record.evidence.counterexamples == ["run_002"]
    assert record.statistics.support == 1
    assert record.statistics.failures == 1
    assert record.confidence == 0.5
    assert record.claim.type.value == "procedure"


def test_merge_drops_candidates_with_zero_supporting_runs() -> None:
    candidates = [
        CandidateEvidence(
            kind="repeated_failure_mode",
            hypothesis="Never ever seen failure",
            mentioned_tools=[],
            mentioned_errors=["nonexistent error"],
        )
    ]
    records = merge(candidates, _evidence_traces())
    assert records == []


def test_merge_success_kind_without_mentioned_tools_needs_successful_runs() -> None:
    candidates = [
        CandidateEvidence(
            kind="repeated_success_strategy",
            hypothesis="Finish with a final answer",
            mentioned_tools=[],
            mentioned_errors=[],
        )
    ]
    records = merge(candidates, _evidence_traces())
    # success kinds: supporting runs must succeed; run_001 succeeded.
    assert len(records) == 1
    assert records[0].evidence.supporting_runs == ["run_001"]
    assert records[0].evidence.counterexamples == ["run_002"]
