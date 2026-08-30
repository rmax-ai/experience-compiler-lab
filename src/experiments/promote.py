"""Candidate evaluation recording + promotion (SPEC.md §10/§11, AGENTS.md §2).

Promotion is the ONLY mutation path for ``skills/``: a candidate that passes
the promotion policy (``evals.policy.decide``, a pure function) is applied
via ``skills.patch.apply_patch`` plus a PURPOSE.yaml provenance bump. Every
decision — accept or reject — is appended to the permanent, append-only
ledger ``results/proposal-history.md`` and written into both the candidate's
``record.yaml`` and its ``results/proposals/<id>.yaml`` copy. Rejected
candidates leave ``skills/`` untouched (rollback is deferred to M5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from evals.policy import decide, decision_reason
from evals.runner import SkillEvalResult, evaluate_candidate
from experiments.proposal_store import ProposalStore
from experiments.runner import ModelFactory
from skills.loader import SKILLS_DIR
from skills.patch import apply_patch
from traces.schema import Scenario, load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[2]

_LEDGER_HEADER = """# Proposal history

Append-only ledger of promotion decisions (SPEC.md §11). Entries are only ever
appended — rejected proposals stay in the history forever.

| candidate | date | decision | versions | success | regressions | reason |
| --- | --- | --- | --- | --- | --- | --- |
"""


class PromotionError(Exception):
    """Raised for double-promote attempts or unevaluated candidates."""


class PromotionOutcome(BaseModel):
    """Result of one promote() call: the decision plus the eval it was based on."""

    model_config = ConfigDict(extra="forbid")

    decision: str  # "accepted" | "rejected"
    eval_result: SkillEvalResult


def load_skill_md(skills_dir: str | Path, workflow: str) -> str:
    """Read ``<skills_dir>/<workflow>/SKILL.md`` (raises FileNotFoundError)."""
    path = Path(skills_dir) / workflow / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill not found: {path}")
    return path.read_text(encoding="utf-8")


def load_skill_version(skills_dir: str | Path, workflow: str) -> str:
    """Read the ``version`` field of ``<skills_dir>/<workflow>/PURPOSE.yaml``."""
    path = Path(skills_dir) / workflow / "PURPOSE.yaml"
    if not path.exists():
        raise FileNotFoundError(f"skill metadata not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    version = data.get("version")
    return str(version) if version is not None else "unknown"


def evaluate_candidate_record(
    candidate_id: str,
    *,
    model_factory: ModelFactory,
    seed: int = 42,
    base_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    validation_scenarios: list[Scenario] | None = None,
    store: TraceStore | None = None,
) -> SkillEvalResult:
    """Evaluate a pending candidate on the validation split; record the result.

    Loads the candidate artifacts, applies the patch to a COPY of the current
    skill, runs ``evals.runner.evaluate_candidate`` against the fixed
    validation split (never the held-out test split), and writes the
    evaluation fields (previous_score, candidate_score, regressions,
    score_vector deltas) into ``record.yaml`` and its
    ``results/proposals/<id>.yaml`` copy (docs/data-formats.md §6).
    """
    root = Path(base_dir) if base_dir is not None else REPO_ROOT
    skills_root = Path(skills_dir) if skills_dir is not None else SKILLS_DIR
    candidate = ProposalStore(root).load_candidate(candidate_id)
    record = candidate["record"]
    workflow = str(record.get("skill"))

    current_skill_md = load_skill_md(skills_root, workflow)
    candidate_skill_md = apply_patch(current_skill_md, str(candidate["patch"]))
    scenarios = validation_scenarios
    if scenarios is None:
        scenarios = load_scenarios(str(root / "datasets" / "validation.jsonl"))

    result = evaluate_candidate(
        baseline_skill_md=current_skill_md,
        candidate_skill_md=candidate_skill_md,
        validation_scenarios=scenarios,
        workflow=workflow,
        model_factory=model_factory,
        seed=seed,
        store=store or TraceStore(root),
    )

    record["evaluation"] = {
        "previous_score": result.baseline_success_rate,
        "candidate_score": result.candidate_success_rate,
        "regressions": len(result.regressions),
        "score_vector": {
            f"{key}_delta": value for key, value in result.score_vector_delta.items()
        },
    }
    _write_record(root, candidate_id, record)
    return result


def promote(
    candidate_id: str,
    *,
    allowed_regressions: int = 0,
    base_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
) -> PromotionOutcome:
    """Decide a candidate's fate and, iff accepted, deploy it.

    Requires the candidate's ``record.yaml`` to carry evaluation fields (run
    :func:`evaluate_candidate_record` first — the ``exp promote`` CLI does this
    automatically when they are missing). Raises :class:`PromotionError` on a
    double-promote attempt or an unevaluated candidate. On accept this is the
    ONLY code path that mutates ``skills/``: SKILL.md via
    ``skills.patch.apply_patch`` and PURPOSE.yaml version/provenance bump.
    """
    root = Path(base_dir) if base_dir is not None else REPO_ROOT
    skills_root = Path(skills_dir) if skills_dir is not None else SKILLS_DIR
    candidate = ProposalStore(root).load_candidate(candidate_id)
    record = candidate["record"]

    prior_decision = record.get("decision")
    if prior_decision not in (None, "pending"):
        raise PromotionError(
            f"candidate {candidate_id} already decided: {prior_decision} "
            "(double-promote protection)"
        )

    evaluation = record.get("evaluation") or {}
    previous_score = evaluation.get("previous_score")
    candidate_score = evaluation.get("candidate_score")
    if previous_score is None or candidate_score is None:
        raise PromotionError(f"candidate {candidate_id} has not been evaluated")

    # The record stores the regression COUNT (docs/data-formats.md §6); the
    # policy only needs rates plus the count, so the result is reconstructed
    # with placeholder regression ids of the recorded length.
    regression_count = int(evaluation.get("regressions") or 0)
    eval_result = SkillEvalResult(
        baseline_success_rate=float(previous_score),
        candidate_success_rate=float(candidate_score),
        regressions=[f"regression-{index + 1}" for index in range(regression_count)],
        score_vector_delta={
            str(key)[: -len("_delta")]: float(value)
            for key, value in (evaluation.get("score_vector") or {}).items()
            if str(key).endswith("_delta")
        },
    )

    accepted = decide(eval_result, allowed_regressions)
    decision = "accepted" if accepted else "rejected"

    record["decision"] = decision
    # AGENTS.md §4 mandates datetime.now(timezone.utc) over the datetime.UTC alias.
    record["decided_at"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    _write_record(root, candidate_id, record)
    _append_ledger(root, record, decision, eval_result, allowed_regressions)

    if accepted:
        workflow = str(record.get("skill"))
        current_skill_md = load_skill_md(skills_root, workflow)
        new_skill_md = apply_patch(current_skill_md, str(candidate["patch"]))
        (skills_root / workflow / "SKILL.md").write_text(new_skill_md, encoding="utf-8")
        _bump_purpose(skills_root, workflow, record)

    return PromotionOutcome(decision=decision, eval_result=eval_result)


def _write_record(root: Path, candidate_id: str, record: dict[str, Any]) -> None:
    """Write record.yaml AND its results/proposals/<id>.yaml copy (same content)."""
    text = yaml.safe_dump(record, sort_keys=True) + "\n"
    candidate_record = root / "results" / "candidates" / candidate_id / "record.yaml"
    candidate_record.write_text(text, encoding="utf-8")
    proposals_dir = root / "results" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    (proposals_dir / f"{candidate_id}.yaml").write_text(text, encoding="utf-8")


def _append_ledger(
    root: Path,
    record: dict[str, Any],
    decision: str,
    eval_result: SkillEvalResult,
    allowed_regressions: int,
) -> None:
    """Append ONE entry to results/proposal-history.md (append-only, never rewritten)."""
    ledger_path = root / "results" / "proposal-history.md"
    if not ledger_path.exists():
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(_LEDGER_HEADER, encoding="utf-8")
    decided = str(record.get("decided_at") or "")[:10]  # YYYY-MM-DD
    entry = (
        f"| {record.get('candidate_id')} | {decided} | "
        f"{'ACCEPT' if decision == 'accepted' else 'REJECT'} | "
        f"v{record.get('from_version')} -> v{record.get('to_version')} | "
        f"{eval_result.baseline_success_rate:.2f} -> "
        f"{eval_result.candidate_success_rate:.2f} | "
        f"regressions {len(eval_result.regressions)} | "
        f"{decision_reason(eval_result, allowed_regressions)} |\n"
    )
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def _bump_purpose(skills_root: Path, workflow: str, record: dict[str, Any]) -> None:
    """PURPOSE.yaml provenance bump on accept: version +1, evidence, evaluation."""
    path = skills_root / workflow / "PURPOSE.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        version = int(str(record.get("from_version"))) + 1
    except (TypeError, ValueError):
        version = data.get("version")
    evaluation = record.get("evaluation") or {}
    data["version"] = version
    data["status"] = "accepted"
    data["derived_from"] = list(record.get("evidence_refs") or [])
    data["evaluation"] = {
        "previous_score": evaluation.get("previous_score"),
        "candidate_score": evaluation.get("candidate_score"),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=True) + "\n", encoding="utf-8")
