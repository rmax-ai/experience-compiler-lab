"""Evolution loop (SPEC.md §11): dev runs -> mine -> propose -> evaluate -> promote.

One iteration: run the dev split with the current skill, mine evidence from
the NEW traces only, propose one minimal patch, evaluate the candidate against
the current skill on the fixed validation split, and apply the promotion
policy. Rejected proposals stay in the history forever — otherwise iteration 8
may repeat the mistake from iteration 3 (SPEC.md §11). The loop stops early
when the proposer finds nothing worth changing.

Splits: dev = datasets/train.jsonl, validation = datasets/validation.jsonl.
The held-out test split is NEVER read by this module (SPEC.md §10; enforced
by a source grep test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evals.runner import run_tasks_with_skill
from experiments.promote import (
    evaluate_candidate_record,
    load_skill_md,
    load_skill_version,
    promote,
)
from experiments.proposal_store import ProposalStore
from experiments.runner import ModelFactory
from knowledge.miner import merge as merge_evidence
from knowledge.miner import mine as mine_evidence
from knowledge.miner import summarize_trace
from knowledge.store import KnowledgeStore
from skills.loader import SKILLS_DIR
from skills.proposer import propose as propose_patch
from traces.schema import Scenario, load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[2]


class IterationRecord(BaseModel):
    """One evolution-loop iteration (SPEC.md §17 report block)."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    runs_created: int
    runs_succeeded: int
    new_record_ids: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    decision: str | None = None
    eval: dict[str, Any] | None = None
    lines_modified: int | None = None


class ProvenanceLink(BaseModel):
    """failure runs -> evidence records -> candidate -> eval -> decision."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    candidate_id: str
    record_ids: list[str] = Field(default_factory=list)
    failure_run_ids: list[str] = Field(default_factory=list)
    previous_score: float
    candidate_score: float
    decision: str


class EvolutionResult(BaseModel):
    """Full evolution-run outcome; the iteration report renders this only."""

    model_config = ConfigDict(extra="forbid")

    workflow: str
    seed: int
    iterations: list[IterationRecord] = Field(default_factory=list)
    provenance: list[ProvenanceLink] = Field(default_factory=list)


def evolve(
    workflow: str,
    iterations: int,
    model_factory: ModelFactory,
    seed: int,
    dev_limit: int = 0,
    validation_scenarios_override: list[Scenario] | None = None,
    *,
    base_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    miner_model_factory: ModelFactory | None = None,
    proposer_model_factory: ModelFactory | None = None,
) -> EvolutionResult:
    """Run the SPEC §11 evolution loop for ``iterations`` iterations.

    ``model_factory`` builds the execution model (fresh per task);
    ``miner_model_factory``/``proposer_model_factory`` default to it and exist
    so scripted fakes can serve each learning-time role its own script. Dev
    runs use per-iteration base seed ``seed + (iteration - 1) * 1000``; the
    candidate evaluation reuses the same iteration seed so baseline and
    candidate see identical per-task seeds (SPEC §10 fairness).
    """
    root = Path(base_dir) if base_dir is not None else REPO_ROOT
    skills_root = Path(skills_dir) if skills_dir is not None else SKILLS_DIR
    miner_factory = miner_model_factory or model_factory
    proposer_factory = proposer_model_factory or model_factory

    dev_scenarios = load_scenarios(str(root / "datasets" / "train.jsonl"))
    if dev_limit > 0:
        dev_scenarios = dev_scenarios[:dev_limit]
    validation = validation_scenarios_override
    if validation is None:
        validation = load_scenarios(str(root / "datasets" / "validation.jsonl"))

    trace_store = TraceStore(root)
    knowledge = KnowledgeStore(root)
    proposals = ProposalStore(root)
    current_skill_md = load_skill_md(skills_root, workflow)

    result = EvolutionResult(workflow=workflow, seed=seed)
    for iteration in range(1, iterations + 1):
        iter_seed = seed + (iteration - 1) * 1000

        # (a) dev runs with the current skill (explicit text, not disk, so an
        # accepted iteration takes effect immediately even in isolated tests).
        traces = run_tasks_with_skill(
            dev_scenarios,
            workflow,
            model_factory,
            iter_seed,
            current_skill_md,
            experiment_id=f"{workflow}-evolve-{seed}-{iteration}",
            store=trace_store,
        )
        runs_succeeded = sum(1 for trace in traces if trace.outcome.success)

        # (b) mine evidence from the NEW traces only; upsert; collect new ids.
        known_ids = {record.id for record in knowledge.all_records()}
        mined = mine_evidence(traces, miner_factory())
        merged = merge_evidence(mined.candidates, traces)
        for record in merged:
            knowledge.upsert(record)
        knowledge.regenerate_index()
        new_record_ids = sorted(record.id for record in merged if record.id not in known_ids)

        # (c) propose from current skill + active records + history + failures.
        active_records = [
            record for record in knowledge.all_records() if record.status == "active"
        ]
        failed_summaries = [
            summarize_trace(trace) for trace in traces if not trace.outcome.success
        ]
        proposal_result = propose_patch(
            skill_md=current_skill_md,
            records=active_records,
            history=proposals.load_history(),
            run_summaries=failed_summaries,
            model=proposer_factory(),
        )

        if proposal_result.proposal is None:
            result.iterations.append(
                IterationRecord(
                    iteration=iteration,
                    runs_created=len(traces),
                    runs_succeeded=runs_succeeded,
                    new_record_ids=new_record_ids,
                )
            )
            break

        # (d) save candidate, evaluate vs current skill on validation, promote.
        proposal = proposal_result.proposal
        candidate_id = proposals.next_candidate_id()
        from_version = load_skill_version(skills_root, workflow)
        proposals.save_candidate(
            candidate_id=candidate_id,
            proposal=proposal,
            workflow=workflow,
            from_version=from_version,
            to_version=_bump_version(from_version),
            proposed_model=str(getattr(model_factory(), "model", "unknown")),
        )
        eval_result = evaluate_candidate_record(
            candidate_id,
            model_factory=model_factory,
            seed=iter_seed,
            base_dir=root,
            skills_dir=skills_root,
            validation_scenarios=validation,
            store=trace_store,
        )
        outcome = promote(
            candidate_id, allowed_regressions=0, base_dir=root, skills_dir=skills_root
        )

        if outcome.decision == "accepted":
            current_skill_md = load_skill_md(skills_root, workflow)

        lines_modified = sum(
            1
            for line in proposal.patch.splitlines()
            if line.startswith("- ") or line.startswith("+ ")
        )
        result.iterations.append(
            IterationRecord(
                iteration=iteration,
                runs_created=len(traces),
                runs_succeeded=runs_succeeded,
                new_record_ids=new_record_ids,
                candidate_id=candidate_id,
                decision=outcome.decision,
                eval={
                    "from_version": from_version,
                    "to_version": _bump_version(from_version),
                    "previous_score": eval_result.baseline_success_rate,
                    "candidate_score": eval_result.candidate_success_rate,
                    "regressions": len(eval_result.regressions),
                    # latency_s is wall-clock and excluded so EvolutionResult
                    # stays byte-deterministic given a deterministic model.
                    "score_vector": {
                        key: value
                        for key, value in eval_result.score_vector_delta.items()
                        if key != "latency_s"
                    },
                },
                lines_modified=lines_modified,
            )
        )
        result.provenance.append(
            ProvenanceLink(
                iteration=iteration,
                candidate_id=candidate_id,
                record_ids=new_record_ids,
                failure_run_ids=[
                    trace.run_id for trace in traces if not trace.outcome.success
                ],
                previous_score=eval_result.baseline_success_rate,
                candidate_score=eval_result.candidate_success_rate,
                decision=outcome.decision,
            )
        )

    return result


def _bump_version(version: str) -> str:
    """Candidate to_version: current version + 1 (best-effort)."""
    try:
        return str(int(version) + 1)
    except (TypeError, ValueError):
        return version
