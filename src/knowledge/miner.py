"""Evidence miner (SPEC.md §7): compact trace summaries → candidate evidence.

The LLM proposes hypotheses; everything else is deterministic. Evidence
links, statistics and confidence are computed from the traces alone, so
identical inputs always produce identical :class:`KnowledgeRecord` objects
(AGENTS.md §5). The execution agent never sees this module (H3) — learning
components consume traces, never the other way around.
"""

from __future__ import annotations

import json
import re
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.adapter import FakeModel, LlmAdapter
from knowledge.schema import (
    Claim,
    ClaimType,
    Evidence,
    KnowledgeRecord,
    Scope,
    Statistics,
    normalize_id,
)
from traces.schema import Trace

# At most this many trace summaries fit into one miner prompt; older runs are
# dropped first (summaries are ordered by run_id, so "most recent" wins).
_MAX_SUMMARIES = 200


class CandidateKind(StrEnum):
    """The eight candidate-evidence kinds the miner prompt asks for (SPEC §7)."""

    repeated_failure_mode = "repeated_failure_mode"
    repeated_success_strategy = "repeated_success_strategy"
    incorrect_assumption = "incorrect_assumption"
    missing_check = "missing_check"
    bad_tool_sequence = "bad_tool_sequence"
    unnecessary_calls = "unnecessary_calls"
    termination_failure = "termination_failure"
    recovery_strategy = "recovery_strategy"


class CandidateEvidence(BaseModel):
    """One structured hypothesis emitted by the miner LLM."""

    model_config = ConfigDict(extra="forbid")

    kind: CandidateKind
    hypothesis: str
    mentioned_tools: list[str] = Field(default_factory=list)
    mentioned_errors: list[str] = Field(default_factory=list)


class MinerResult(BaseModel):
    """Outcome of one mine() call (candidates plus usage accounting)."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateEvidence] = Field(default_factory=list)
    parse_failures: int = 0
    model_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


MINER_PROMPT = """You are an evidence miner. You receive a compact JSON array of summaries of agent execution runs. Identify recurring patterns:
- repeated_failure_mode: a failure that recurs across runs
- repeated_success_strategy: a successful approach that recurs across runs
- incorrect_assumption: the agent assumed something false about the world
- missing_check: a world state the agent should have checked before acting
- bad_tool_sequence: tools invoked in the wrong order
- unnecessary_calls: tool calls that wasted steps
- termination_failure: the agent stopped too early or never terminated cleanly
- recovery_strategy: a way the agent recovered from a failure

Respond with STRICT JSON only: a JSON array of candidate objects, each with EXACTLY these fields:
{"kind": "<one of the eight kinds above>", "hypothesis": "<short actionable statement>", "mentioned_tools": ["<tool name>", ...], "mentioned_errors": ["<error substring>", ...]}

No prose outside the JSON, no markdown fences. If you find nothing, output an empty array []."""

_REPAIR_PROMPT = (
    "Your previous response was not valid JSON and could not be parsed. "
    "Respond with ONLY a valid JSON array of candidate objects (or [] if none). "
    "No prose, no markdown fences.\n\nRaw response:\n{raw}"
)

# A markdown fenced block, optionally tagged with a language (e.g. ```json).
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)


def summarize_trace(trace: Trace) -> dict[str, Any]:
    """Compact, deterministic run summary for the miner prompt.

    Errors and violated constraints are sorted; tool calls keep their
    execution order. ``error_key`` is the raw tool error string for failed
    calls, empty for successful ones.
    """
    tool_sequence = [
        {
            "tool": action.tool,
            "ok": action.result.get("ok") is True,
            "error_key": action.result.get("error") if action.result.get("ok") is False else "",
        }
        for action in trace.actions
    ]
    return {
        "run_id": trace.run_id,
        "task_id": trace.task_id,
        "success": trace.outcome.success,
        "errors": sorted(trace.outcome.errors),
        "violated_constraints": sorted(trace.outcome.violated_constraints),
        "tool_sequence": tool_sequence,
        "tool_call_count": len(trace.actions),
    }


def mine(traces: list[Trace], model: LlmAdapter | FakeModel) -> MinerResult:
    """Mine a batch of traces into candidate evidence.

    One user message carries the JSON array of trace summaries (at most
    ``_MAX_SUMMARIES``; the most recent by run_id order). The response must be
    a JSON array of :class:`CandidateEvidence`; markdown fences are stripped.
    Invalid items are dropped and counted; a whole-payload parse failure is
    retried once with a repair prompt, and counted if it fails again.
    """
    summaries = [summarize_trace(t) for t in sorted(traces, key=lambda t: t.run_id)]
    if len(summaries) > _MAX_SUMMARIES:
        summaries = summaries[-_MAX_SUMMARIES:]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MINER_PROMPT},
        {"role": "user", "content": json.dumps(summaries, sort_keys=True)},
    ]

    model_calls = 0
    tokens = 0
    cost_usd = 0.0
    parse_failures = 0

    result = model.complete(messages)
    model_calls += 1
    tokens += result.usage.input_tokens + result.usage.output_tokens
    cost_usd += result.usage.estimated_cost_usd
    raw = str(result.message.get("content") or "")

    candidates, dropped, payload_failed = _parse_candidates(raw)
    parse_failures += dropped
    if payload_failed:
        parse_failures += 1
        repair = messages + [{"role": "user", "content": _REPAIR_PROMPT.format(raw=raw)}]
        result = model.complete(repair)
        model_calls += 1
        tokens += result.usage.input_tokens + result.usage.output_tokens
        cost_usd += result.usage.estimated_cost_usd
        raw = str(result.message.get("content") or "")
        candidates, dropped, payload_failed = _parse_candidates(raw)
        parse_failures += dropped
        if payload_failed:
            parse_failures += 1
            candidates = []

    return MinerResult(
        candidates=candidates,
        parse_failures=parse_failures,
        model_calls=model_calls,
        tokens=tokens,
        cost_usd=cost_usd,
    )


def _parse_candidates(raw: str) -> tuple[list[CandidateEvidence], int, bool]:
    """Parse the model response.

    Returns ``(valid candidates, dropped invalid items, whole-payload failed)``.
    A non-list or unparseable payload is a whole-payload failure; individual
    objects that fail validation are dropped and counted.
    """
    text = _strip_markdown_fences(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], 0, True
    if not isinstance(data, list):
        return [], 0, True
    candidates: list[CandidateEvidence] = []
    dropped = 0
    for item in data:
        try:
            candidates.append(CandidateEvidence.model_validate(item))
        except ValidationError:
            dropped += 1
    return candidates, dropped, False


def _strip_markdown_fences(text: str) -> str:
    """Extract a ```json ... ``` (or plain ``` ... ```) block if present.

    Falls back to the whole text when no fence is found, so the JSON parser
    sees the raw response and can trigger the repair retry on garbage.
    """
    match = _FENCE_RE.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


_FAILURE_KINDS = frozenset(
    {
        CandidateKind.repeated_failure_mode,
        CandidateKind.incorrect_assumption,
        CandidateKind.missing_check,
        CandidateKind.bad_tool_sequence,
        CandidateKind.unnecessary_calls,
        CandidateKind.termination_failure,
    }
)
_SUCCESS_KINDS = frozenset(
    {
        CandidateKind.repeated_success_strategy,
        CandidateKind.recovery_strategy,
    }
)


def merge(candidates: list[CandidateEvidence], traces: list[Trace]) -> list[KnowledgeRecord]:
    """Deterministically merge candidate hypotheses into knowledge records.

    Evidence links are computed purely from the traces:

    - supporting runs for failure kinds: runs where any action-result error
      contains one of ``mentioned_errors``; if ``mentioned_errors`` is empty,
      runs that contain at least one failed action with an error.
    - supporting runs for success kinds: runs that succeeded AND used every
      ``mentioned_tool`` at least once.
    - counterexamples for failure kinds: successful runs that used all
      ``mentioned_tools``.
    - counterexamples for success kinds: failed runs.

    ``statistics.support = len(supporting_runs)`` and
    ``statistics.failures = len(counterexamples)``; confidence is the
    Laplace-smoothed share of supporting evidence,
    ``(support + 1) / (support + failures + 2)``, which never collapses to
    0 or 1 on small samples.

    Candidates with zero supporting runs are dropped (they are not evidence).
    Candidates that normalize to the same id are merged (evidence is only ever
    combined, never removed) and the result is sorted by id.
    """
    errors_by_run: dict[str, set[str]] = {}
    tools_by_run: dict[str, set[str]] = {}
    succeeded: dict[str, bool] = {}
    for trace in traces:
        run_errors: set[str] = set()
        run_tools: set[str] = set()
        for action in trace.actions:
            run_tools.add(action.tool)
            error = action.result.get("error")
            if isinstance(error, str) and error:
                run_errors.add(error)
        errors_by_run[trace.run_id] = run_errors
        tools_by_run[trace.run_id] = run_tools
        succeeded[trace.run_id] = trace.outcome.success

    grouped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        supporting, counterexamples = _evidence_links(
            candidate, traces, errors_by_run, tools_by_run, succeeded
        )
        if not supporting:
            continue
        key = normalize_id(candidate.hypothesis)
        entry = grouped.setdefault(
            key, {"candidate": candidate, "supporting": set(), "counterexamples": set()}
        )
        entry["supporting"].update(supporting)
        entry["counterexamples"].update(counterexamples)

    today = date.today()
    records: list[KnowledgeRecord] = []
    for key in sorted(grouped):
        entry = grouped[key]
        supporting = sorted(entry["supporting"])
        counterexamples = sorted(entry["counterexamples"])
        support = len(supporting)
        failures = len(counterexamples)
        records.append(
            KnowledgeRecord(
                id=key,
                claim=Claim(
                    type=_claim_type(entry["candidate"].kind),
                    text=entry["candidate"].hypothesis,
                ),
                scope=Scope(workflows=["onboarding"]),
                evidence=Evidence(supporting_runs=supporting, counterexamples=counterexamples),
                statistics=Statistics(support=support, failures=failures),
                confidence=(support + 1) / (support + failures + 2),
                status="active",
                first_seen=today,
                last_updated=today,
                format_version=1,
            )
        )
    return records


def _evidence_links(
    candidate: CandidateEvidence,
    traces: list[Trace],
    errors_by_run: dict[str, set[str]],
    tools_by_run: dict[str, set[str]],
    succeeded: dict[str, bool],
) -> tuple[list[str], list[str]]:
    """Compute supporting and counterexample run ids for one candidate."""
    run_ids = [trace.run_id for trace in traces]
    if candidate.kind in _FAILURE_KINDS:
        mentioned = candidate.mentioned_errors
        if mentioned:
            supporting = [
                run_id
                for run_id in run_ids
                if any(
                    any(err in error for err in mentioned)
                    for error in errors_by_run[run_id]
                )
            ]
        else:
            supporting = [run_id for run_id in run_ids if errors_by_run[run_id]]
        counterexamples = [
            run_id
            for run_id in run_ids
            if succeeded[run_id]
            and all(tool in tools_by_run[run_id] for tool in candidate.mentioned_tools)
        ]
    else:  # success kinds
        supporting = [
            run_id
            for run_id in run_ids
            if succeeded[run_id]
            and all(tool in tools_by_run[run_id] for tool in candidate.mentioned_tools)
        ]
        counterexamples = [run_id for run_id in run_ids if not succeeded[run_id]]
    return supporting, counterexamples


def _claim_type(kind: CandidateKind) -> ClaimType:
    """Claim type mapping: success/recovery strategies are procedures;
    every failure kind is a warning."""
    if kind in _SUCCESS_KINDS:
        return ClaimType.procedure
    return ClaimType.warning
