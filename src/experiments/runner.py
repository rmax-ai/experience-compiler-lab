"""Experiment runner: scenarios -> traces with graded outcomes (SPEC.md §3).

Owns the per-run orchestration that the executor deliberately does not: run_id
allocation, manifest assembly (git short hashes, model, seed), skill version,
and the deterministic grader decision. The runner never touches the
``knowledge`` package either (H3) — learning components consume traces, never
the other way around.
"""

from __future__ import annotations

from collections.abc import Callable

from agent.adapter import FakeModel, LlmAdapter
from agent.context import build_context
from agent.executor import Executor, build_world
from evals.graders import evaluate
from skills.loader import get_skill_version
from traces.schema import Manifest, Outcome, Scenario, Trace, git_short_hash
from traces.store import TraceStore
from world.state import World

ModelFactory = Callable[[], LlmAdapter | FakeModel]


def run_tasks(
    scenarios: list[Scenario],
    workflow: str,
    model_factory: ModelFactory,
    seed: int,
    experiment_id: str,
    store: TraceStore | None = None,
    persist: bool = True,
    memory_notes: list[str] | None = None,
) -> list[Trace]:
    """Run every scenario and append the graded traces to the store.

    Per-task seed is ``seed + index`` (deterministic). Each task gets a fresh
    world rebuilt from its scenario block and a fresh model from
    ``model_factory``, so scripted/stateful models never leak across tasks.
    ``Outcome.success`` is decided by the deterministic grader; the executor's
    transport-level errors (e.g. step-budget exhaustion) are preserved.
    """
    store = store or TraceStore()
    traces: list[Trace] = []
    for index, scenario in enumerate(scenarios):
        task_seed = seed + index
        world = build_world(scenario)
        model = model_factory()
        executor = Executor(model=model, world=world)
        run_id = store.next_run_id()

        manifest = Manifest(
            experiment_id=experiment_id,
            model={"name": model.model, "temperature": getattr(model, "temperature", 0.0)},
            dataset_version=git_short_hash(),
            skill_version=git_short_hash(),
            knowledge_version=git_short_hash(),
            environment_version=git_short_hash(),
            seed=task_seed,
        )

        context = (
            build_context(scenario, workflow=workflow)
            if memory_notes is None
            else build_context(scenario, workflow=workflow, memory_notes=memory_notes)
        )
        trace = executor.run(
            task=scenario,
            context=context,
            seed=task_seed,
            run_id=run_id,
            manifest=manifest,
            skill_version=get_skill_version(workflow),
        )

        trace.outcome = _grade(world, scenario, trace)
        if persist:
            store.append(trace)
        traces.append(trace)
    return traces


def _grade(world: World, scenario: Scenario, trace: Trace) -> Outcome:
    """Map the deterministic grader result onto the trace, keeping transport errors."""
    graded = evaluate(world, scenario.grader)
    transport_errors = list(trace.outcome.errors)
    return Outcome(
        success=graded.success,
        errors=sorted(set(graded.errors + transport_errors)),
        violated_constraints=sorted(set(graded.violated_constraints)),
    )
