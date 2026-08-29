"""Proposer tests: JSON parsing, repair retry, patch validation, citation extraction."""

import json
from datetime import date

from agent.adapter import FakeModel
from knowledge.schema import Claim, ClaimType, Evidence, KnowledgeRecord, Scope, Statistics
from skills.proposer import (
    PROPOSER_PROMPT,
    Proposal,
    ProposerResult,
    extract_cited_records,
    propose,
)

SKILL = """# Employee Onboarding

## Procedure
1. Resolve employee identity.
2. Retrieve onboarding requirements.
3. Check hardware inventory.
4. Assign available hardware.
"""

_VALID_PATCH = "@@ Procedure\n- 4. Assign available hardware.\n+ 4. Assign hardware only if inventory confirms availability.\n"


def _record(
    record_id: str,
    *,
    confidence: float = 0.8,
    support: int = 4,
    failures: int = 1,
) -> KnowledgeRecord:
    return KnowledgeRecord(
        id=record_id,
        claim=Claim(type=ClaimType.procedure, text=f"claim for {record_id}"),
        scope=Scope(workflows=["onboarding"]),
        evidence=Evidence(supporting_runs=["run_001"], counterexamples=["run_002"]),
        statistics=Statistics(support=support, failures=failures),
        confidence=confidence,
        status="active",
        first_seen=date(2026, 8, 1),
        last_updated=date(2026, 8, 1),
    )


def _script(content: str, usage: dict | None = None) -> tuple[dict, dict]:
    return (
        {"role": "assistant", "content": content},
        usage or {"input_tokens": 50, "output_tokens": 30},
    )


def test_propose_parses_valid_payload() -> None:
    payload = {
        "reasoning": "records: inventory-check-before-assignment",
        "patch": _VALID_PATCH,
    }
    model = FakeModel(scripts=[_script(json.dumps(payload))])
    result = propose(
        SKILL,
        [_record("inventory-check-before-assignment")],
        history=[],
        run_summaries=[],
        model=model,
    )
    assert isinstance(result, ProposerResult)
    assert result.proposal is not None
    assert result.proposal.reasoning == "records: inventory-check-before-assignment"
    assert result.proposal.patch == _VALID_PATCH
    assert result.proposal.cited_records == ["inventory-check-before-assignment"]
    assert result.parse_failures == 0
    assert result.model_calls == 1
    assert result.tokens == 80
    assert result.cost_usd > 0.0


def test_propose_strips_markdown_fences() -> None:
    payload = {"reasoning": "ok", "patch": _VALID_PATCH}
    prose = f"Here you go:\n```json\n{json.dumps(payload)}\n```"
    model = FakeModel(scripts=[_script(prose)])
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is not None
    assert result.parse_failures == 0


def test_propose_repair_retry_recovers_valid_json() -> None:
    payload = {"reasoning": "recovered", "patch": _VALID_PATCH}
    model = FakeModel(
        scripts=[_script("oops I wrote prose"), _script(json.dumps(payload))]
    )
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is not None
    assert result.proposal.reasoning == "recovered"
    assert result.parse_failures == 1
    assert result.model_calls == 2


def test_propose_garbage_twice_returns_empty() -> None:
    model = FakeModel(scripts=[_script("this is not json"), _script("still not json {{{")])
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is None
    assert result.parse_failures == 2
    assert result.model_calls == 2


def test_propose_empty_patch_means_no_proposal() -> None:
    payload = {"reasoning": "nothing worth changing", "patch": ""}
    model = FakeModel(scripts=[_script(json.dumps(payload))])
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is None
    assert result.parse_failures == 0
    assert result.model_calls == 1


def test_propose_drops_patch_referencing_nonexistent_section() -> None:
    payload = {"reasoning": "oops", "patch": "@@ NoSuchSection\n- x.\n+ y.\n"}
    model = FakeModel(scripts=[_script(json.dumps(payload))])
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is None
    assert result.parse_failures == 1  # dropped, never retried, never auto-fixed
    assert result.model_calls == 1


def test_propose_drops_patch_that_does_not_apply() -> None:
    payload = {"reasoning": "oops", "patch": "@@ Procedure\n- this line is missing.\n+ x.\n"}
    model = FakeModel(scripts=[_script(json.dumps(payload))])
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is None
    assert result.parse_failures == 1
    assert result.model_calls == 1


def test_propose_non_dict_payload_retries() -> None:
    model = FakeModel(
        scripts=[
            _script(json.dumps(["not", "an", "object"])),
            _script(json.dumps({"reasoning": "fixed", "patch": _VALID_PATCH})),
        ]
    )
    result = propose(SKILL, [], [], [], model)
    assert result.proposal is not None
    assert result.parse_failures == 1
    assert result.model_calls == 2


# -- extract_cited_records -----------------------------------------------------


def test_extract_cited_records_deterministic() -> None:
    reasoning = (
        "records: inventory-check-before-assignment and "
        "look-up-the-employee-record-before-granting-access"
    )
    record_ids = [
        "inventory-check-before-assignment",
        "look-up-the-employee-record-before-granting-access",
        "unrelated-claim",
    ]
    first = extract_cited_records(reasoning, record_ids)
    second = extract_cited_records(reasoning, record_ids)
    assert first == [
        "inventory-check-before-assignment",
        "look-up-the-employee-record-before-granting-access",
    ]
    assert first == second


def test_extract_cited_records_matches_normalized_form() -> None:
    # The id stored in the record may carry non-slug characters; the citation
    # check uses its normalized form.
    reasoning = "evidence supports inventory-check-before-assignment here"
    assert extract_cited_records(reasoning, ["Inventory Check Before Assignment"]) == [
        "Inventory Check Before Assignment"
    ]


def test_extract_cited_records_requires_whole_token() -> None:
    assert extract_cited_records("inventory-check-before-assignment-2", ["inventory-check-before-assignment"]) == []
    assert extract_cited_records("pre-inventory-check-before-assignment", ["inventory-check-before-assignment"]) == []
    assert extract_cited_records("inventory-check-before-assignment", ["inventory-check-before-assignment"]) == [
        "inventory-check-before-assignment"
    ]


def test_extract_cited_records_empty_inputs() -> None:
    assert extract_cited_records("", ["inventory-check-before-assignment"]) == []
    assert extract_cited_records("any text", []) == []


# -- prompt --------------------------------------------------------------------


def test_proposer_prompt_is_single_string() -> None:
    assert isinstance(PROPOSER_PROMPT, str)
    assert "STRICT JSON" in PROPOSER_PROMPT
    assert "patch" in PROPOSER_PROMPT


def test_proposal_model_shape() -> None:
    proposal = Proposal(reasoning="r", patch=_VALID_PATCH, cited_records=["a-claim"])
    assert proposal.model_dump() == {
        "reasoning": "r",
        "patch": _VALID_PATCH,
        "cited_records": ["a-claim"],
    }
