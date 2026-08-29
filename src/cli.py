"""experience-compiler-lab CLI (thin dispatch layer; commands live in subpackages)."""

import os
from pathlib import Path

import typer

from agent.adapter import FakeModel, LlmAdapter
from experiments.runner import ModelFactory, run_tasks
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


if __name__ == "__main__":
    app()
