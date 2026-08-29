"""KnowledgeRecord schema tests: YAML roundtrip, normalize_id, extra rejection."""

from datetime import date

import pytest
from pydantic import ValidationError

from knowledge.schema import (
    Claim,
    ClaimType,
    Evidence,
    KnowledgeRecord,
    Scope,
    Statistics,
    from_yaml,
    normalize_id,
    to_yaml,
)


def _record(**overrides: object) -> KnowledgeRecord:
    base = KnowledgeRecord(
        id="check-inventory-before-assignment",
        claim=Claim(
            type=ClaimType.procedure,
            text="Check available inventory before assigning hardware.",
        ),
        scope=Scope(workflows=["onboarding"]),
        evidence=Evidence(supporting_runs=["run_014", "run_031"], counterexamples=["run_022"]),
        statistics=Statistics(support=8, failures=2),
        confidence=0.82,
        status="active",
        first_seen=date(2026, 8, 29),
        last_updated=date(2026, 8, 29),
    )
    return base.model_copy(update=overrides)


def test_yaml_roundtrip_lossless() -> None:
    record = _record(
        supersedes=["older-claim"],
        superseded_by=None,
    )
    assert from_yaml(to_yaml(record)) == record


def test_yaml_roundtrip_with_supersession_links() -> None:
    record = _record(
        status="superseded",
        supersedes=["older-claim"],
        superseded_by=["newer-claim"],
    )
    assert from_yaml(to_yaml(record)) == record


def test_normalize_id_lowercases_and_collapses_spaces() -> None:
    assert normalize_id("Check Inventory Before Assignment") == "check-inventory-before-assignment"
    assert normalize_id("check   inventory    before") == "check-inventory-before"


def test_normalize_id_collapses_punctuation_runs() -> None:
    assert normalize_id("check!! inventory?? before...") == "check-inventory-before"
    assert normalize_id("  --leading and trailing--  ") == "leading-and-trailing"


def test_normalize_id_truncates_beyond_64_chars() -> None:
    slug = normalize_id("x" * 100)
    assert len(slug) == 64
    assert slug == "x" * 64
    long = normalize_id(" ".join(["inventory"] * 20))
    assert len(long) <= 64


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        KnowledgeRecord(
            id="extra-field",
            claim=Claim(type=ClaimType.warning, text="nope"),
            scope=Scope(workflows=["onboarding"]),
            evidence=Evidence(),
            statistics=Statistics(support=0, failures=0),
            confidence=0.5,
            first_seen=date(2026, 8, 29),
            last_updated=date(2026, 8, 29),
            unexpected=True,
        )


def test_nested_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Claim(type=ClaimType.procedure, text="x", surprise="y")
    with pytest.raises(ValidationError):
        Statistics(support=1, failures=0, extra=2)
