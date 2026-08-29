"""experience-compiler-lab CLI (thin dispatch layer; commands live in subpackages)."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from agent.adapter import FakeModel, LlmAdapter
from experiments.runner import ModelFactory, run_tasks
from knowledge.miner import merge as merge_evidence
from knowledge.miner import mine as mine_evidence
from knowledge.store import KnowledgeStore
from traces.schema import load_scenarios
from traces.store import TraceStore

app = typer.Typer(name="exp", help="Agent Experience Compiler")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
DEV_SCRIPTS = REPO_ROOT / "src" / "agent" / "fixtures" / "dev.jsonl"


@app.callback()
def main() -> None:
    """Agent Experience Compiler — root group.

    NOTE: the callback exists so the app is ALWAYS built as a click Group.
    Typer 0.27 collapses single-command apps into that command, which breaks
    CliRunner tests (`invoke(app, ["version"])` → "unexpected extra
    arguments") and changes behavior as soon as a second command is added.
    """


@app.command()
def version() -> None:
    """Print version."""
    typer.echo("0.1.0")


@app.command()
def run(
    split: str = typer.Argument(
        "train", help="dataset split: train | validation | test"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="max scenarios to run (default: all)"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="model name (default: fake unless EXP_LLM_API_KEY is set)",
    ),
    workflow: str = typer.Option(
        "onboarding", "--workflow", help="skill workflow directory"
    ),
    seed: int = typer.Option(42, "--seed", help="base run seed"),
) -> None:
    """Run scenarios against the active skill and write traces to experience/runs/."""
    if split not in ("train", "validation", "test"):
        raise typer.BadParameter("split must be one of: train, validation, test")
    scenarios = load_scenarios(str(DATASETS_DIR / f"{split}.jsonl"))
    if limit is not None:
        scenarios = scenarios[:limit]

    model_name = _resolve_model_name(model)
    traces = run_tasks(
        scenarios=scenarios,
        workflow=workflow,
        model_factory=_model_factory(model_name),
        seed=seed,
        experiment_id=f"{workflow}-{split}-{seed}",
    )
    for trace in traces:
        typer.echo(
            f"{trace.run_id} {trace.task_id} success={trace.outcome.success} "
            f"tools={trace.metrics.tool_calls} cost={trace.metrics.estimated_cost_usd:.6f}"
        )


@app.command()
def inspect(run_id: str) -> None:
    """Print a compact summary of one run."""
    store = TraceStore()
    try:
        trace = store.get(run_id)
    except FileNotFoundError:
        typer.echo(f"run not found: {run_id}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"run_id: {trace.run_id}")
    typer.echo(f"task: {trace.task_id}")
    typer.echo(f"model: {trace.model}")
    typer.echo(f"success: {trace.outcome.success}")
    typer.echo(f"errors: {trace.outcome.errors}")
    typer.echo(f"violated_constraints: {trace.outcome.violated_constraints}")
    typer.echo(f"tool_calls: {trace.metrics.tool_calls}")
    typer.echo(f"tokens: in={trace.metrics.tokens_in} out={trace.metrics.tokens_out}")
    typer.echo(f"cost_usd: {trace.metrics.estimated_cost_usd:.6f}")
    if trace.final_answer is not None:
        typer.echo(f"final_answer: {trace.final_answer}")


@app.command()
def mine(
    since_run: str | None = typer.Option(
        None, "--since-run", help="only mine runs with run_id >= this (default: all)"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="max runs to mine (default: all)"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="model name (default: fake unless EXP_LLM_API_KEY is set)",
    ),
) -> None:
    """Mine traces into knowledge records under knowledge/patterns/."""
    store = TraceStore()
    rows = store.list_runs()
    if since_run is not None:
        rows = [row for row in rows if row["run_id"] >= since_run]
    if limit is not None:
        rows = rows[:limit]
    traces = [store.get(row["run_id"]) for row in rows]

    model_name = _resolve_model_name(model)
    result = mine_evidence(traces, _miner_model_factory(model_name)())
    records = merge_evidence(result.candidates, traces)

    knowledge = KnowledgeStore()
    for record in records:
        upserted = knowledge.upsert(record)
        typer.echo(
            f"upserted {upserted.id} support={upserted.statistics.support} "
            f"failures={upserted.statistics.failures} confidence={upserted.confidence:.2f}"
        )
    knowledge.regenerate_index()
    typer.echo(f"records: {len(records)}")


@app.command()
def patterns() -> None:
    """List knowledge records."""
    for record in KnowledgeStore().all_records():
        typer.echo(
            f"{record.id} {record.status} support={record.statistics.support} "
            f"confidence={record.confidence:.2f}"
        )


def _resolve_model_name(override: str | None) -> str:
    """Default model: fake when no API key, otherwise EXP_LLM_MODEL."""
    if override is not None:
        return override
    if os.environ.get("EXP_LLM_API_KEY"):
        return os.environ.get("EXP_LLM_MODEL", "fake")
    return "fake"


def _model_factory(model_name: str) -> ModelFactory:
    """Factory building a fresh model per task (scripted fake or live adapter)."""
    if model_name == "fake" or not os.environ.get("EXP_LLM_API_KEY"):

        def factory() -> FakeModel:
            return FakeModel(scripts=DEV_SCRIPTS, model=model_name, temperature=0.0)

        return factory

    base_url = os.environ.get("EXP_LLM_BASE_URL")
    api_key = os.environ.get("EXP_LLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "EXP_LLM_BASE_URL and EXP_LLM_API_KEY must be set to run a real model"
        )

    def factory() -> LlmAdapter:
        return LlmAdapter(base_url=base_url, api_key=api_key, model=model_name, temperature=0.0)

    return factory


# Scripted candidates served by the fake miner (deterministic dev path): one
# failure-mode hypothesis keyed to inventory errors and one success-strategy
# hypothesis keyed to the employee lookup before granting access.
_MINER_FAKE_CANDIDATES: list[dict[str, Any]] = [
    {
        "kind": "repeated_failure_mode",
        "hypothesis": "Check available inventory before assigning hardware",
        "mentioned_tools": ["assign_device"],
        "mentioned_errors": ["inventory error"],
    },
    {
        "kind": "repeated_success_strategy",
        "hypothesis": "Look up the employee record before granting access",
        "mentioned_tools": ["get_employee", "grant_access"],
        "mentioned_errors": [],
    },
]

_MINER_FAKE_USAGE: dict[str, int] = {"input_tokens": 96, "output_tokens": 64}


def _miner_model_factory(model_name: str) -> Callable[[], LlmAdapter | FakeModel]:
    """Factory building a fresh model per mine() call (fake or live adapter)."""
    if model_name == "fake" or not os.environ.get("EXP_LLM_API_KEY"):

        def factory() -> FakeModel:
            response = {"role": "assistant", "content": json.dumps(_MINER_FAKE_CANDIDATES)}
            # Two copies so the JSON-repair retry also succeeds under the fake.
            scripts = [(response, dict(_MINER_FAKE_USAGE)) for _ in range(2)]
            return FakeModel(scripts=scripts, model=model_name, temperature=0.0)

        return factory

    base_url = os.environ.get("EXP_LLM_BASE_URL")
    api_key = os.environ.get("EXP_LLM_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "EXP_LLM_BASE_URL and EXP_LLM_API_KEY must be set to run a real model"
        )

    def factory() -> LlmAdapter:
        return LlmAdapter(base_url=base_url, api_key=api_key, model=model_name, temperature=0.0)

    return factory


if __name__ == "__main__":
    app()
