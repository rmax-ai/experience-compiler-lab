"""Knowledge record contract (docs/data-formats.md §3, SPEC.md §6).

Structured, append-only interpretations distilled from execution traces.
The miner emits :class:`KnowledgeRecord` objects; the store persists them as
``knowledge/patterns/<id>.yaml`` and never removes evidence (AGENTS.md §2).
All models forbid extra fields so schema drift fails loudly at IO boundaries.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ClaimType(StrEnum):
    """Semantic category of a claim (docs/data-formats.md §3)."""

    procedure = "procedure"
    assumption = "assumption"
    constraint = "constraint"
    warning = "warning"


class Claim(BaseModel):
    """The claim itself: a category plus the human-readable statement."""

    model_config = ConfigDict(extra="forbid")

    type: ClaimType
    text: str


class Scope(BaseModel):
    """Workflows a claim applies to."""

    model_config = ConfigDict(extra="forbid")

    workflows: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """Run ids backing (or contradicting) a claim — append-only sets."""

    model_config = ConfigDict(extra="forbid")

    supporting_runs: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)


class Statistics(BaseModel):
    """Aggregate counts derived deterministically from :class:`Evidence`."""

    model_config = ConfigDict(extra="forbid")

    support: int
    failures: int


class KnowledgeRecord(BaseModel):
    """One persistent knowledge record (docs/data-formats.md §3).

    ``first_seen``/``last_updated`` are calendar dates only (no wall-clock
    timestamps — documented contract); ``format_version`` is the artifact
    format version, bumped on breaking schema changes.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    claim: Claim
    scope: Scope
    evidence: Evidence
    statistics: Statistics
    confidence: float
    status: Literal["active", "superseded"] = "active"
    first_seen: date
    last_updated: date
    supersedes: list[str] | None = None
    superseded_by: list[str] | None = None
    format_version: int = 1


def normalize_id(text: str) -> str:
    """Deterministic record id: lowercase, non-alphanumeric runs collapse to
    a single hyphen, edges stripped, at most 64 characters."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:64]


def to_yaml(record: KnowledgeRecord) -> str:
    """Serialize a record to YAML (sort_keys=True → byte-deterministic)."""
    return yaml.safe_dump(record.model_dump(mode="json"), sort_keys=True)


def from_yaml(text: str) -> KnowledgeRecord:
    """Parse a record from YAML text."""
    return KnowledgeRecord.model_validate(yaml.safe_load(text))
