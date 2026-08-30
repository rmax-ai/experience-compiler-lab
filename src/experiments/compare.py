"""M5 held-out ablation runs (SPEC.md §12 and §19)."""

from __future__ import annotations

import contextlib
import csv
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

import experiments.runner as _runner
from agent.context import build_context
from experiments.configs import AblationConfig
from experiments.evolution import evolve
from experiments.promote import evaluate_candidate_record, load_skill_version, promote
from experiments.proposal_store import ProposalStore
from experiments.runner import ModelFactory, run_tasks
from knowledge.miner import merge, mine, summarize_trace
from knowledge.store import KnowledgeStore
from skills.loader import SKILLS_DIR
from skills.proposer import propose
from traces.schema import Scenario, Trace, load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[2]


class PerConfigResult(BaseModel):
    """Comparable aggregate from one M5 configuration."""

    model_config = ConfigDict(extra="forbid")

    config: str
    dev_success_rate: float | None = None
    validation_success_rate: float | None = None
    heldout_success_rate: float
    heldout_successes: int
    heldout_total: int
    avg_tool_calls: float
    avg_tokens: int
    avg_cost_usd: float
    skill_version: int
    runs_created: int
    records_created: int
    decisions: list[str] = Field(default_factory=list)


def run_config(
    config: str,
    workflow: str,
    iterations: int,
    model_factory: ModelFactory,
    seed: int,
    dev_limit: int = 0,
) -> PerConfigResult:
    """Run one configuration while preserving append-only stores.

    Only this module opens ``datasets/test.jsonl``.  All task execution uses
    the same base seed, and ``run_tasks`` derives its per-task seeds from
    scenario order, making held-out comparisons fair.
    """
    try:
        selected = AblationConfig(config)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in AblationConfig)
        raise ValueError(f"unknown compare config {config!r}; expected one of {allowed}") from exc

    store = TraceStore(REPO_ROOT)
    knowledge = KnowledgeStore(REPO_ROOT)
    run_start = _run_number(store.next_run_id())
    record_ids_before = {record.id for record in knowledge.all_records()}
    dev = load_scenarios(str(REPO_ROOT / "datasets" / "train.jsonl"))
    if dev_limit > 0:
        dev = dev[:dev_limit]
    validation = load_scenarios(str(REPO_ROOT / "datasets" / "validation.jsonl"))
    heldout = load_scenarios(str(REPO_ROOT / "datasets" / "test.jsonl"))

    dev_rate: float | None = None
    validation_rate: float | None = None
    decisions: list[str] = []

    if selected is AblationConfig.BASELINE:
        heldout_traces = _run_with_skill(
            heldout,
            workflow,
            model_factory,
            seed,
            "",
            f"compare-{config}-{seed}",
            store,
            persist=True,
        )
        skill_version = 0
    elif selected is AblationConfig.TRACE2SKILL:
        heldout_traces, dev_rate, validation_rate, decisions = _trace2skill(
            dev, validation, heldout, workflow, iterations, model_factory, seed, store
        )
        skill_version = _version_as_int(load_skill_version(SKILLS_DIR, workflow))
    elif selected is AblationConfig.MEMORY:
        heldout_traces, dev_rate = _memory(
            dev, heldout, workflow, iterations, model_factory, seed, store, knowledge
        )
        skill_version = _version_as_int(load_skill_version(SKILLS_DIR, workflow))
    else:
        evolution = evolve(workflow, iterations, model_factory, seed, dev_limit)
        decisions = [
            record.decision for record in evolution.iterations if record.decision is not None
        ]
        evaluated = [record.eval for record in evolution.iterations if record.eval is not None]
        if evaluated:
            validation_rate = float(evaluated[-1]["candidate_score"])
        if evolution.iterations:
            last = evolution.iterations[-1]
            dev_rate = _rate(last.runs_succeeded, last.runs_created)
        heldout_traces = run_tasks(
            heldout,
            workflow,
            model_factory,
            seed,
            f"compare-{config}-{seed}",
            store,
            persist=True,
        )
        skill_version = _version_as_int(load_skill_version(SKILLS_DIR, workflow))

    record_ids_after = {record.id for record in knowledge.all_records()}
    return _result(
        config,
        heldout_traces,
        dev_rate,
        validation_rate,
        skill_version,
        max(0, _run_number(store.next_run_id()) - run_start),
        len(record_ids_after - record_ids_before),
        decisions,
    )


def compare(
    configs: list[str],
    workflow: str,
    iterations: int,
    model_factory: ModelFactory,
    seed: int,
    dev_limit: int = 0,
) -> list[PerConfigResult]:
    """Run configurations in caller-supplied order with one shared seed."""
    return [
        run_config(config, workflow, iterations, model_factory, seed, dev_limit)
        for config in configs
    ]


def write_compare_report(results: list[PerConfigResult], seed: int) -> tuple[Path, Path]:
    """Write reproducible Markdown and CSV aggregates under ``results/``."""
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    md_path = results_dir / f"compare-{seed}.md"
    csv_path = results_dir / f"compare-{seed}.csv"
    columns = [
        "config",
        "dev",
        "validation",
        "heldout",
        "tool_calls",
        "tokens",
        "cost",
        "skill_version",
        "decisions",
    ]
    rows = [_report_row(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# M5 compare",
        "",
        f"Seed: {seed}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    lines.extend(
        [
            "",
            "## Hypotheses",
            "",
            "- H1: compare Trace2Skill with Compiler to test whether structured knowledge improves proposals.",
            "- H2: compare Compiler with Baseline on held-out success to test whether evolved skills generalize.",
            "- H3: compare Memory with Compiler to test whether keeping knowledge out of executor context yields more reusable skills.",
            "",
            "This run shows deterministic outcomes for these configurations and seed. It cannot establish statistical significance, model transfer, or generalization beyond this simulator and scenario set.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


def _trace2skill(
    dev: list[Scenario],
    validation: list[Scenario],
    heldout: list[Scenario],
    workflow: str,
    iterations: int,
    model_factory: ModelFactory,
    seed: int,
    store: TraceStore,
) -> tuple[list[Trace], float | None, float | None, list[str]]:
    proposals = ProposalStore(REPO_ROOT)
    dev_rate: float | None = None
    validation_rate: float | None = None
    decisions: list[str] = []
    for iteration in range(1, iterations + 1):
        iter_seed = seed + (iteration - 1) * 1000
        traces = run_tasks(
            dev,
            workflow,
            model_factory,
            iter_seed,
            f"compare-trace2skill-{seed}-{iteration}",
            store,
        )
        dev_rate = _rate(sum(trace.outcome.success for trace in traces), len(traces))
        proposal_result = propose(
            skill_md=(SKILLS_DIR / workflow / "SKILL.md").read_text(encoding="utf-8"),
            records=[],
            history=proposals.load_history(),
            run_summaries=[summarize_trace(trace) for trace in traces if not trace.outcome.success],
            model=model_factory(),
        )
        if proposal_result.proposal is None:
            continue
        candidate_id = proposals.next_candidate_id()
        version = load_skill_version(SKILLS_DIR, workflow)
        proposals.save_candidate(
            candidate_id,
            proposal_result.proposal,
            workflow,
            version,
            str(_version_as_int(version) + 1),
            str(model_factory().model),
        )
        evaluation = evaluate_candidate_record(
            candidate_id,
            model_factory=model_factory,
            seed=iter_seed,
            validation_scenarios=validation,
            store=store,
        )
        validation_rate = evaluation.candidate_success_rate
        outcome = promote(candidate_id, allowed_regressions=0)
        decisions.append(outcome.decision)
    heldout_traces = run_tasks(
        heldout, workflow, model_factory, seed, f"compare-trace2skill-{seed}", store
    )
    return heldout_traces, dev_rate, validation_rate, decisions


def _memory(
    dev: list[Scenario],
    heldout: list[Scenario],
    workflow: str,
    iterations: int,
    model_factory: ModelFactory,
    seed: int,
    store: TraceStore,
    knowledge: KnowledgeStore,
) -> tuple[list[Trace], float | None]:
    dev_rate: float | None = None
    for iteration in range(1, iterations + 1):
        notes = _memory_notes(knowledge)
        traces = run_tasks(
            dev,
            workflow,
            model_factory,
            seed + (iteration - 1) * 1000,
            f"compare-memory-{seed}-{iteration}",
            store,
            memory_notes=notes,
        )
        dev_rate = _rate(sum(trace.outcome.success for trace in traces), len(traces))
        mined = mine(traces, model_factory())
        for record in merge(mined.candidates, traces):
            knowledge.upsert(record)
        knowledge.regenerate_index()
    heldout_traces = run_tasks(
        heldout,
        workflow,
        model_factory,
        seed,
        f"compare-memory-{seed}",
        store,
        memory_notes=_memory_notes(knowledge),
    )
    return heldout_traces, dev_rate


@contextlib.contextmanager
def _empty_skill() -> Iterator[None]:
    original = _runner.build_context
    _runner.build_context = lambda task, workflow="onboarding", memory_notes=None: build_context(
        task, workflow, skill_loader=lambda _workflow: "", memory_notes=memory_notes
    )
    try:
        yield
    finally:
        _runner.build_context = original


def _run_with_skill(
    scenarios: list[Scenario],
    workflow: str,
    model_factory: ModelFactory,
    seed: int,
    skill_md: str,
    experiment_id: str,
    store: TraceStore,
    *,
    persist: bool,
) -> list[Trace]:
    if skill_md:
        return run_tasks(
            scenarios, workflow, model_factory, seed, experiment_id, store, persist=persist
        )
    with _empty_skill():
        return run_tasks(
            scenarios, workflow, model_factory, seed, experiment_id, store, persist=persist
        )


def _memory_notes(knowledge: KnowledgeStore) -> list[str]:
    records = [record for record in knowledge.all_records() if record.status == "active"]
    return [
        f"{record.id}: {record.claim.text}"
        for record in sorted(records, key=lambda r: (-r.confidence, r.id))[:8]
    ]


def _result(
    config: str,
    traces: list[Trace],
    dev_rate: float | None,
    validation_rate: float | None,
    skill_version: int,
    runs_created: int,
    records_created: int,
    decisions: list[str],
) -> PerConfigResult:
    total = len(traces)
    return PerConfigResult(
        config=config,
        dev_success_rate=dev_rate,
        validation_success_rate=validation_rate,
        heldout_success_rate=_rate(sum(trace.outcome.success for trace in traces), total) or 0.0,
        heldout_successes=sum(trace.outcome.success for trace in traces),
        heldout_total=total,
        avg_tool_calls=sum(trace.metrics.tool_calls for trace in traces) / total if total else 0.0,
        avg_tokens=round(
            sum(trace.metrics.tokens_in + trace.metrics.tokens_out for trace in traces) / total
        )
        if total
        else 0,
        avg_cost_usd=sum(trace.metrics.estimated_cost_usd for trace in traces) / total
        if total
        else 0.0,
        skill_version=skill_version,
        runs_created=runs_created,
        records_created=records_created,
        decisions=decisions,
    )


def _report_row(result: PerConfigResult) -> dict[str, str | int]:
    return {
        "config": result.config,
        "dev": _percent(result.dev_success_rate),
        "validation": _percent(result.validation_success_rate),
        "heldout": _percent(result.heldout_success_rate),
        "tool_calls": f"{result.avg_tool_calls:.2f}",
        "tokens": result.avg_tokens,
        "cost": f"{result.avg_cost_usd:.6f}",
        "skill_version": result.skill_version,
        "decisions": ",".join(result.decisions),
    }


def _rate(successes: int, total: int) -> float | None:
    return round(successes / total, 4) if total else None


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _run_number(run_id: str) -> int:
    return int(run_id.removeprefix("run_"))


def _version_as_int(version: str) -> int:
    try:
        return int(version)
    except ValueError:
        return 0
