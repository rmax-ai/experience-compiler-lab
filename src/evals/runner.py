"""Candidate skill evaluation harness (SPEC.md §10, docs/data-formats.md §6).

Evaluates a candidate skill against the SAME fixed validation split as the
current skill, with identical per-task seeds (``seed + index``) and identical
scenario order on both sides, so the comparison is fair. The held-out test
split is NEVER touched here — only the validation scenarios handed in by the
caller.

Both sides go through ``experiments.runner.run_tasks`` (runs are appended to
the TraceStore — they are evidence). ``run_tasks`` loads the skill from disk,
so to evaluate an *unpromoted* candidate the context builder is temporarily
pointed at the explicit skill text for the duration of the call and restored
afterwards (the harness is single-threaded).

Determinism note: given a deterministic model the result is deterministic
except ``latency_s`` (wall-clock). Scripted models must therefore serve the
identical script sequence for both sides — ``model_factory`` is called fresh
per task on each side, so a stateless script source yields a fair comparison.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field

import experiments.runner as _runner
from agent.context import build_context
from experiments.runner import ModelFactory, run_tasks
from traces.schema import Scenario, Trace
from traces.store import TraceStore

_SCORE_KEYS = ("tool_calls", "tokens", "cost_usd", "latency_s", "trajectory_length")


class SkillEvalResult(BaseModel):
    """Baseline-vs-candidate comparison on the fixed validation split."""

    model_config = ConfigDict(extra="forbid")

    baseline_success_rate: float
    candidate_success_rate: float
    regressions: list[str] = Field(default_factory=list)
    score_vector_delta: dict[str, float] = Field(default_factory=dict)
    baseline_by_task: dict[str, bool] = Field(default_factory=dict)
    candidate_by_task: dict[str, bool] = Field(default_factory=dict)


@contextlib.contextmanager
def _skill_override(skill_md: str) -> Iterator[None]:
    """Point ``experiments.runner``'s context builder at an explicit skill text.

    ``run_tasks`` calls ``build_context(scenario, workflow=workflow)``, which
    reads ``skills/<workflow>/SKILL.md`` from disk; evaluating an unpromoted
    candidate requires injecting the skill text instead. The swap is restored
    on exit, so a failed evaluation never leaks into later runs.
    """
    original = _runner.build_context

    def _build_with_skill(task: Scenario, workflow: str = "onboarding"):  # noqa: ANN202
        return build_context(task, workflow=workflow, skill_loader=lambda _w: skill_md)

    _runner.build_context = _build_with_skill
    try:
        yield
    finally:
        _runner.build_context = original


def run_tasks_with_skill(
    scenarios: list[Scenario],
    workflow: str,
    model_factory: ModelFactory,
    seed: int,
    skill_md: str,
    experiment_id: str,
    store: TraceStore | None = None,
) -> list[Trace]:
    """``run_tasks`` with an explicit skill text instead of the deployed skill."""
    with _skill_override(skill_md):
        return run_tasks(
            scenarios=scenarios,
            workflow=workflow,
            model_factory=model_factory,
            seed=seed,
            experiment_id=experiment_id,
            store=store,
        )


def evaluate_candidate(
    baseline_skill_md: str,
    candidate_skill_md: str,
    validation_scenarios: list[Scenario],
    workflow: str,
    model_factory: ModelFactory,
    seed: int,
    *,
    store: TraceStore | None = None,
) -> SkillEvalResult:
    """Compare baseline and candidate skills on the fixed validation split.

    Both sides run every scenario in the SAME order with the SAME per-task
    seed (``seed + index``), each task in a fresh world rebuilt from the
    scenario block (``run_tasks`` semantics). Success per task comes from the
    deterministic grader via the returned traces' ``outcome.success``.
    """
    if not validation_scenarios:
        raise ValueError("validation_scenarios must be non-empty")
    task_ids = [scenario.task_id for scenario in validation_scenarios]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"duplicate task_id values in validation scenarios: {task_ids}")

    baseline_traces = run_tasks_with_skill(
        validation_scenarios,
        workflow,
        model_factory,
        seed,
        baseline_skill_md,
        experiment_id=f"{workflow}-eval-baseline-{seed}",
        store=store,
    )
    candidate_traces = run_tasks_with_skill(
        validation_scenarios,
        workflow,
        model_factory,
        seed,
        candidate_skill_md,
        experiment_id=f"{workflow}-eval-candidate-{seed}",
        store=store,
    )

    total = len(validation_scenarios)
    baseline_by_task = {trace.task_id: trace.outcome.success for trace in baseline_traces}
    candidate_by_task = {trace.task_id: trace.outcome.success for trace in candidate_traces}
    regressions = sorted(
        task_id
        for task_id in task_ids
        if baseline_by_task[task_id] and not candidate_by_task[task_id]
    )

    baseline_totals = _score_totals(baseline_traces)
    candidate_totals = _score_totals(candidate_traces)
    score_vector_delta = {
        key: round(candidate_totals[key] - baseline_totals[key], 8) for key in _SCORE_KEYS
    }

    return SkillEvalResult(
        baseline_success_rate=round(sum(baseline_by_task.values()) / total, 4),
        candidate_success_rate=round(sum(candidate_by_task.values()) / total, 4),
        regressions=regressions,
        score_vector_delta=score_vector_delta,
        baseline_by_task=baseline_by_task,
        candidate_by_task=candidate_by_task,
    )


def _score_totals(traces: list[Trace]) -> dict[str, float]:
    """Sum the SPEC §10 score-vector metrics across one side's traces."""
    return {
        "tool_calls": float(sum(trace.metrics.tool_calls for trace in traces)),
        "tokens": float(
            sum(trace.metrics.tokens_in + trace.metrics.tokens_out for trace in traces)
        ),
        "cost_usd": float(sum(trace.metrics.estimated_cost_usd for trace in traces)),
        "latency_s": float(sum(trace.metrics.latency_s for trace in traces)),
        "trajectory_length": float(sum(trace.metrics.trajectory_length for trace in traces)),
    }
