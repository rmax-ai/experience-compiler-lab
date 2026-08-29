"""Skill proposer (SPEC.md §9, docs/data-formats.md §5/§6).

The proposer turns structured knowledge, recent run summaries and proposal
history into ONE minimal, applyable skill patch. It never writes to
``skills/<workflow>/`` directly — the M3 isolation rule: candidate artifacts
land under ``results/candidates/<id>/`` and only promotion (M4) mutates
deployed skills. A patch that does not apply cleanly to the current skill is
dropped and counted, never auto-fixed, so every candidate is a clean diff.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.adapter import FakeModel, LlmAdapter
from knowledge.schema import KnowledgeRecord, normalize_id
from skills.patch import verify_applies

_MAX_RECORDS = 50
_MAX_RUNS = 20
_MAX_HISTORY = 20

PROPOSER_PROMPT = """You improve a procedural skill for an enterprise-tools agent. Input: the current skill markdown, structured knowledge records (each with id, claim text, type, support, failures, confidence), a summary of recent failed runs (task, errors, violated constraints), and recent proposal history (patch summaries and decisions). Propose ONLY ONE minimal patch fixing the most valuable single improvement. Output STRICT JSON with exactly: {"reasoning": string (cite record ids when used), "patch": string (diff using the patch grammar; lines starting "@@ ", "- ", "+ " only; the section header names must exist in the current skill)}. If nothing is worth changing, output {"reasoning": "...", "patch": ""}."""

_REPAIR_PROMPT = (
    "Your previous response was not valid JSON (expected a JSON object with "
    '"reasoning" and "patch" string fields). Respond with ONLY that strict '
    'JSON object; use "patch": "" when nothing is worth changing. '
    "No prose, no markdown fences.\n\nRaw response:\n{raw}"
)

_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)


class Proposal(BaseModel):
    """One proposed skill change: the reasoning, the patch, and cited records."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    patch: str
    cited_records: list[str] = Field(default_factory=list)


class ProposerResult(BaseModel):
    """Outcome of one propose() call (proposal plus usage accounting)."""

    model_config = ConfigDict(extra="forbid")

    proposal: Proposal | None = None
    model_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    parse_failures: int = 0


def extract_cited_records(reasoning: str, record_ids: list[str]) -> list[str]:
    """Return the record ids cited in ``reasoning`` (deterministic).

    A record id is cited when its normalized form (``knowledge.schema.
    normalize_id``) appears in the reasoning as a complete token — never as a
    prefix or suffix of a longer id-like token. Results keep the order of
    ``record_ids`` and are deduplicated.
    """
    cited: list[str] = []
    for record_id in record_ids:
        normalized = normalize_id(record_id)
        if not normalized:
            continue
        if _cited_pattern(normalized).search(reasoning) and record_id not in cited:
            cited.append(record_id)
    return cited


def propose(
    skill_md: str,
    records: list[KnowledgeRecord],
    history: list[dict[str, Any]],
    run_summaries: list[dict[str, Any]],
    model: LlmAdapter | FakeModel,
) -> ProposerResult:
    """Propose ONE minimal patch from knowledge, runs, and proposal history.

    Inputs are capped deterministically: records by confidence descending
    (at most 50), the most recent runs and history entries (at most 20 each).
    The model response must be strict JSON ``{"reasoning": str, "patch":
    str}``; markdown fences are stripped. A whole-payload parse failure is
    retried once with a repair prompt. A parsed patch that does not apply to
    the current skill is dropped and counted as a parse failure — never
    retried, never auto-fixed. An empty ``patch`` string means "nothing worth
    changing" and yields no proposal.
    """
    records = sorted(records, key=lambda record: (-record.confidence, record.id))[:_MAX_RECORDS]
    runs = sorted(run_summaries, key=lambda summary: str(summary.get("run_id", "")))[-_MAX_RUNS:]
    history = sorted(history, key=lambda entry: str(entry.get("candidate_id", "")))[
        -_MAX_HISTORY:
    ]

    payload = {
        "skill": skill_md,
        "records": [
            {
                "id": record.id,
                "claim": record.claim.text,
                "type": record.claim.type.value,
                "support": record.statistics.support,
                "failures": record.statistics.failures,
                "confidence": record.confidence,
            }
            for record in records
        ],
        "run_summaries": runs,
        "history": history,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROPOSER_PROMPT},
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]
    record_ids = [record.id for record in records]

    model_calls = 0
    tokens = 0
    cost_usd = 0.0
    parse_failures = 0

    result = model.complete(messages)
    model_calls += 1
    tokens += result.usage.input_tokens + result.usage.output_tokens
    cost_usd += result.usage.estimated_cost_usd
    raw = str(result.message.get("content") or "")

    proposal, payload_failed, patch_rejected = _parse_proposal(raw, skill_md, record_ids)
    if patch_rejected:
        parse_failures += 1
    if payload_failed:
        parse_failures += 1
        repair = messages + [{"role": "user", "content": _REPAIR_PROMPT.format(raw=raw)}]
        result = model.complete(repair)
        model_calls += 1
        tokens += result.usage.input_tokens + result.usage.output_tokens
        cost_usd += result.usage.estimated_cost_usd
        raw = str(result.message.get("content") or "")
        proposal, payload_failed, patch_rejected = _parse_proposal(raw, skill_md, record_ids)
        if patch_rejected:
            parse_failures += 1
        if payload_failed:
            parse_failures += 1
            proposal = None

    return ProposerResult(
        proposal=proposal,
        model_calls=model_calls,
        tokens=tokens,
        cost_usd=cost_usd,
        parse_failures=parse_failures,
    )


def _parse_proposal(
    raw: str, skill_md: str, record_ids: list[str]
) -> tuple[Proposal | None, bool, bool]:
    """Parse one model response.

    Returns ``(proposal, payload_failed, patch_rejected)``. A payload that is
    not a JSON object with string ``reasoning`` and ``patch`` fields is a
    whole-payload failure (triggers the repair retry). A valid payload whose
    patch does not apply to the current skill is ``patch_rejected`` — dropped
    and counted, never retried, never auto-fixed.
    """
    text = _strip_markdown_fences(raw)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, True, False
    if not isinstance(data, dict):
        return None, True, False
    reasoning = data.get("reasoning")
    patch = data.get("patch")
    if not isinstance(reasoning, str) or not isinstance(patch, str):
        return None, True, False
    if not patch:
        return None, False, False  # valid "nothing worth changing" response
    if not verify_applies(skill_md, patch):
        return None, False, True
    return (
        Proposal(
            reasoning=reasoning,
            patch=patch,
            cited_records=extract_cited_records(reasoning, record_ids),
        ),
        False,
        False,
    )


def _strip_markdown_fences(text: str) -> str:
    """Extract a ```json ... ``` (or plain ``` ... ```) block if present.

    Falls back to the whole text when no fence is found, so the JSON parser
    sees the raw response and can trigger the repair retry on garbage.
    """
    match = _FENCE_RE.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


_CITED_PATTERNS: dict[str, re.Pattern[str]] = {}


def _cited_pattern(normalized: str) -> re.Pattern[str]:
    """Whole-token match for a normalized record id (cached, deterministic).

    The id must be surrounded by characters outside ``[a-z0-9-]`` so it is a
    complete token, not a prefix or suffix of a longer id-like token.
    """
    pattern = _CITED_PATTERNS.get(normalized)
    if pattern is None:
        pattern = re.compile(
            rf"(?<![a-z0-9-]){re.escape(normalized)}(?![a-z0-9-])",
            re.IGNORECASE,
        )
        _CITED_PATTERNS[normalized] = pattern
    return pattern
