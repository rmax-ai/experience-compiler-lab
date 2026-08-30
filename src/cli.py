"""experience-compiler-lab CLI (thin dispatch layer; commands live in subpackages)."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from agent.adapter import FakeModel, LlmAdapter
from evals.policy import decide, decision_reason
from experiments.compare import compare as compare_configs
from experiments.compare import write_compare_report
from experiments.evolution import evolve as evolve_loop
from experiments.matrix import (
    run_transfer_matrix,
    write_transfer_matrix_csv,
    write_transfer_matrix_report,
)
from experiments.promote import (
    PromotionError,
    evaluate_candidate_record,
)
from experiments.promote import (
    promote as promote_candidate,
)
from experiments.proposal_store import ProposalStore
from experiments.report import write_iteration_report
from experiments.runner import ModelFactory, run_tasks
from knowledge.miner import merge as merge_evidence
from knowledge.miner import mine as mine_evidence
from knowledge.miner import summarize_trace
from knowledge.store import KnowledgeStore
from skills.loader import get_skill_version, load_skill
from skills.proposer import propose as propose_patch
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
def inspect(artifact: str) -> None:
    """Print one run (run_<id>), knowledge record (<pattern-id>), or candidate (candidate-<id>)."""
    if artifact.startswith("candidate-"):
        _inspect_candidate(artifact)
    elif artifact.startswith("run_"):
        _inspect_run(artifact)
    else:
        _inspect_pattern(artifact)


@app.command()
def propose(
    workflow: str = typer.Argument("onboarding", help="skill workflow directory"),
    model: str | None = typer.Option(
        None,
        "--model",
        help="model name (default: fake unless EXP_LLM_API_KEY is set)",
    ),
    traces: int = typer.Option(
        20, "--traces", help="max recent runs to summarize for the proposer"
    ),
) -> None:
    """Propose ONE minimal skill patch; write candidates under results/candidates/.

    The proposer NEVER mutates skills/<workflow>/ directly — candidate
    artifacts land in results/candidates/<id>/ and promotion (M4) is the only
    path that changes deployed skills.
    """
    skill_md = load_skill(workflow)
    knowledge = KnowledgeStore()
    records = [record for record in knowledge.all_records() if record.status == "active"]
    history = ProposalStore().load_history()
    run_summaries = _recent_run_summaries(traces)

    model_name = _resolve_model_name(model)
    result = propose_patch(
        skill_md=skill_md,
        records=records,
        history=history,
        run_summaries=run_summaries,
        model=_proposer_model_factory(model_name)(),
    )

    if result.proposal is None:
        typer.echo(f"no patch proposed (parse_failures={result.parse_failures})")
        return

    store = ProposalStore()
    candidate_id = store.next_candidate_id()
    from_version = get_skill_version(workflow)
    store.save_candidate(
        candidate_id=candidate_id,
        proposal=result.proposal,
        workflow=workflow,
        from_version=from_version,
        to_version=_bump_version(from_version),
        proposed_model=model_name,
    )
    patch_lines = len([line for line in result.proposal.patch.splitlines() if line.strip()])
    typer.echo(f"candidate: {candidate_id}")
    typer.echo(f"patch_lines: {patch_lines}")
    typer.echo(f"cited_records: {', '.join(result.proposal.cited_records)}")
    typer.echo(f"cost_usd: {result.cost_usd:.6f}")


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


@app.command(name="eval")
def eval_command(
    candidate_id: str = typer.Argument(..., help="candidate id (candidate-<NN>)"),
    model: str | None = typer.Option(
        None,
        "--model",
        help="model name (default: fake unless EXP_LLM_API_KEY is set)",
    ),
    seed: int = typer.Option(42, "--seed", help="base run seed"),
    allowed_regressions: int = typer.Option(
        0, "--allowed-regressions", help="max tolerated regressions"
    ),
) -> None:
    """Evaluate a candidate against the fixed validation split and record it.

    Baseline (deployed skill) and candidate run the SAME scenarios with the
    SAME per-task seeds. Writes the evaluation fields into the candidate's
    record.yaml and prints the decision preview; the held-out test split is
    never touched.
    """
    model_name = _resolve_model_name(model)
    try:
        result = evaluate_candidate_record(
            candidate_id, model_factory=_model_factory(model_name), seed=seed
        )
    except FileNotFoundError:
        typer.echo(f"candidate not found: {candidate_id}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"baseline_success_rate: {result.baseline_success_rate:.4f}")
    typer.echo(f"candidate_success_rate: {result.candidate_success_rate:.4f}")
    if result.regressions:
        typer.echo(f"regressions: {len(result.regressions)} ({', '.join(result.regressions)})")
    else:
        typer.echo("regressions: 0")
    for key, value in result.score_vector_delta.items():
        typer.echo(f"delta_{key}: {value}")
    accepted = decide(result, allowed_regressions)
    typer.echo(f"decision preview: {'would accept' if accepted else 'would reject'}")
    typer.echo(f"reason: {decision_reason(result, allowed_regressions)}")


@app.command(name="promote")
def promote_command(
    candidate_id: str = typer.Argument(..., help="candidate id (candidate-<NN>)"),
    allowed_regressions: int = typer.Option(
        0, "--allowed-regressions", help="max tolerated regressions"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="model name (default: fake unless EXP_LLM_API_KEY is set)",
    ),
    seed: int = typer.Option(42, "--seed", help="base run seed (used if eval is missing)"),
) -> None:
    """Apply the promotion policy to a candidate; only accept mutates skills/.

    If the candidate has not been evaluated yet, the evaluation runs first
    (no separate `exp eval` call required). The decision is recorded in
    record.yaml, results/proposals/<id>.yaml and the append-only ledger.
    """
    model_name = _resolve_model_name(model)
    try:
        record = ProposalStore().load_candidate(candidate_id)["record"]
    except FileNotFoundError:
        typer.echo(f"candidate not found: {candidate_id}", err=True)
        raise typer.Exit(1) from None
    evaluation = record.get("evaluation") or {}
    if evaluation.get("previous_score") is None or evaluation.get("candidate_score") is None:
        evaluate_candidate_record(
            candidate_id, model_factory=_model_factory(model_name), seed=seed
        )
    try:
        outcome = promote_candidate(candidate_id, allowed_regressions=allowed_regressions)
    except PromotionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"decision: {outcome.decision}")
    typer.echo(f"baseline_success_rate: {outcome.eval_result.baseline_success_rate:.4f}")
    typer.echo(f"candidate_success_rate: {outcome.eval_result.candidate_success_rate:.4f}")
    typer.echo(
        "skills/: updated" if outcome.decision == "accepted" else "skills/: unchanged"
    )


@app.command()
def evolve(
    iterations: int = typer.Option(5, "--iterations", help="evolution iterations"),
    dev_limit: int = typer.Option(
        0, "--dev-limit", help="max dev scenarios per iteration (0 = all)"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="model name (default: fake unless EXP_LLM_API_KEY is set)",
    ),
    seed: int = typer.Option(42, "--seed", help="base run seed"),
    workflow: str = typer.Option("onboarding", "--workflow", help="skill workflow directory"),
) -> None:
    """Run the evolution loop: dev runs -> mine -> propose -> evaluate -> promote."""
    model_name = _resolve_model_name(model)
    result = evolve_loop(
        workflow=workflow,
        iterations=iterations,
        model_factory=_model_factory(model_name),
        seed=seed,
        dev_limit=dev_limit,
        miner_model_factory=_miner_model_factory(model_name),
        proposer_model_factory=_proposer_model_factory(model_name),
    )
    reports_dir = REPO_ROOT / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "iteration-report.md"
    if report_path.exists():
        report_path = reports_dir / f"iteration-{seed}.md"
    write_iteration_report(result, str(report_path))
    for iteration in result.iterations:
        typer.echo(
            f"iteration {iteration.iteration}: runs={iteration.runs_created} "
            f"success={iteration.runs_succeeded} "
            f"new_records={len(iteration.new_record_ids)} "
            f"candidate={iteration.candidate_id or '-'} "
            f"decision={iteration.decision or '-'}"
        )
    typer.echo(f"report: {report_path}")


@app.command()
def compare(
    configs: str = typer.Option(
        "baseline,trace2skill,memory,compiler",
        "--configs",
        help="comma-separated subset: baseline,trace2skill,memory,compiler",
    ),
    iterations: int = typer.Option(3, "--iterations", help="iterations per configuration"),
    dev_limit: int = typer.Option(0, "--dev-limit", help="max dev scenarios (0 = all)"),
    model: str | None = typer.Option(None, "--model", help="execution/learning model"),
    seed: int = typer.Option(42, "--seed", help="base run seed"),
    workflow: str = typer.Option("onboarding", "--workflow", help="skill workflow directory"),
) -> None:
    """Compare M5 ablations and write a held-out report plus CSV."""
    selected = [item.strip() for item in configs.split(",") if item.strip()]
    if not selected:
        raise typer.BadParameter("configs must contain at least one configuration")
    model_name = _resolve_model_name(model)
    results = compare_configs(
        selected, workflow, iterations, _compare_model_factory(model_name), seed, dev_limit
    )
    write_compare_report(results, seed)
    for result in results:
        typer.echo(
            f"config={result.config} heldout={result.heldout_success_rate:.2%} "
            f"skill_v{result.skill_version} decisions={result.decisions}"
        )


@app.command()
def matrix(
    models: str = typer.Option(
        ...,
        "--models",
        help="comma-separated model names (e.g. fake,gpt-x); each is trained and executed",
    ),
    iterations: int = typer.Option(2, "--iterations", help="evolution iterations per source"),
    seed: int = typer.Option(42, "--seed", help="base run seed"),
    dev_limit: int | None = typer.Option(
        None, "--dev-limit", help="max dev scenarios per iteration (default: all)"
    ),
    workflow: str = typer.Option("onboarding", "--workflow", help="skill workflow directory"),
) -> None:
    """Cross-model transfer matrix: train one skill per model, run every skill on every model."""
    selected = [item.strip() for item in models.split(",") if item.strip()]
    if not selected:
        raise typer.BadParameter("models must contain at least one model name")
    result = run_transfer_matrix(
        selected, workflow, iterations, _matrix_model_factory, seed, dev_limit
    )
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_transfer_matrix_csv(result, results_dir / "transfer-matrix.csv")
    md_path = write_transfer_matrix_report(
        result, results_dir / "transfer-matrix.md", iterations=iterations
    )
    for cell in result.cells:
        typer.echo(
            f"{cell.skill_source}->{cell.executor_model} "
            f"heldout={cell.heldout_success_rate:.2%} "
            f"skill_v{cell.skill_version} decisions={','.join(cell.decisions) or '-'} "
            f"cost={cell.cost_usd:.6f}"
        )
    typer.echo(f"csv: {csv_path}")
    typer.echo(f"md: {md_path}")


@app.command()
def report(
    path: str = typer.Option(
        "results/reports/iteration-report.md", "--path", help="report file to print"
    ),
) -> None:
    """Print the path of the latest iteration report."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        typer.echo(f"report not found: {resolved}", err=True)
        raise typer.Exit(1)
    typer.echo(str(resolved))


def _inspect_run(run_id: str) -> None:
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


def _inspect_candidate(candidate_id: str) -> None:
    """Print a candidate's patch, reasoning, evidence refs and evaluation state."""
    store = ProposalStore()
    try:
        candidate = store.load_candidate(candidate_id)
    except FileNotFoundError:
        typer.echo(f"candidate not found: {candidate_id}", err=True)
        raise typer.Exit(1) from None
    record = candidate["record"]
    typer.echo(f"candidate_id: {candidate_id}")
    typer.echo(f"skill: {record.get('skill')}")
    typer.echo(f"from_version: {record.get('from_version')}")
    typer.echo(f"to_version: {record.get('to_version')}")
    typer.echo(f"evidence_refs: {record.get('evidence_refs')}")
    typer.echo(f"evaluation: {record.get('evaluation')}")
    typer.echo(f"decision: {record.get('decision')}")
    typer.echo("reasoning:")
    for line in str(candidate["reasoning"]).splitlines():
        typer.echo(f"  {line}")
    typer.echo("patch:")
    for line in str(candidate["patch"]).splitlines():
        typer.echo(f"  {line}")


def _inspect_pattern(pattern_id: str) -> None:
    """Print one knowledge record."""
    store = KnowledgeStore()
    try:
        record = store.get(pattern_id)
    except (FileNotFoundError, ValueError):
        typer.echo(f"record not found: {pattern_id}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"id: {record.id}")
    typer.echo(f"claim: {record.claim.text}")
    typer.echo(f"type: {record.claim.type.value}")
    typer.echo(f"status: {record.status}")
    typer.echo(f"support: {record.statistics.support}")
    typer.echo(f"failures: {record.statistics.failures}")
    typer.echo(f"confidence: {record.confidence:.2f}")
    typer.echo(f"supporting_runs: {record.evidence.supporting_runs}")
    typer.echo(f"counterexamples: {record.evidence.counterexamples}")


def _recent_run_summaries(limit: int) -> list[dict]:
    """Most recent run summaries in knowledge.miner.summarize_trace shape."""
    if limit <= 0:
        return []
    store = TraceStore()
    rows = store.list_runs()  # ordered by run_id, so the last rows are newest
    return [summarize_trace(store.get(row["run_id"])) for row in rows[-limit:]]


def _bump_version(version: str) -> str:
    """Candidate to_version: current version + 1 (best-effort)."""
    try:
        return str(int(version) + 1)
    except (TypeError, ValueError):
        return version


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


# Scripted proposal served by the fake proposer (deterministic dev path): a
# one-line Procedure change citing the active employee-lookup record, which
# applies cleanly to the current onboarding skill.
_PROPOSER_FAKE_RESPONSE: dict[str, str] = {
    "reasoning": (
        "The active knowledge record look-up-the-employee-record-before-granting-access "
        "supports verifying identity before granting access; assign hardware only when "
        "inventory confirms availability."
    ),
    "patch": "@@ Procedure\n"
    "- 4. Assign available hardware.\n"
    "+ 4. Assign hardware only if inventory confirms availability.\n",
}
_PROPOSER_FAKE_USAGE: dict[str, int] = {"input_tokens": 120, "output_tokens": 64}


def _proposer_model_factory(model_name: str) -> Callable[[], LlmAdapter | FakeModel]:
    """Factory building a fresh model per propose() call (fake or live adapter)."""
    if model_name == "fake" or not os.environ.get("EXP_LLM_API_KEY"):

        def factory() -> FakeModel:
            response = {"role": "assistant", "content": json.dumps(_PROPOSER_FAKE_RESPONSE)}
            # Two copies so the JSON-repair retry also succeeds under the fake.
            scripts = [(response, dict(_PROPOSER_FAKE_USAGE)) for _ in range(2)]
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


def _matrix_model_factory(model_name: str) -> ModelFactory:
    """Per-model factory for the transfer matrix.

    Live adapter when an API key is configured and the model is not the fake;
    otherwise the deterministic role-routing fake (execution, mining and
    proposal scripts served by prompt sniffing).
    """
    if model_name != "fake" and os.environ.get("EXP_LLM_API_KEY"):
        return _model_factory(model_name)
    return _compare_model_factory(model_name)


def _compare_model_factory(model_name: str) -> ModelFactory:
    """Route the deterministic fake's execution, mining and proposal scripts.

    A real adapter needs no routing: its prompt determines the task.  The
    fixture-backed fake has separate deterministic scripts for each role.
    """
    if model_name != "fake" and os.environ.get("EXP_LLM_API_KEY"):
        return _model_factory(model_name)

    class CompareFakeModel:
        model = model_name
        temperature = 0.0

        def __init__(self) -> None:
            self._execution = _model_factory(model_name)()

        def complete(
            self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1024
        ):  # noqa: ANN201
            system = str(messages[0].get("content", "")) if messages else ""
            if "You are an evidence miner" in system:
                return _miner_model_factory(model_name)().complete(messages, tools, max_tokens)
            if "You improve a procedural skill" in system:
                return _proposer_model_factory(model_name)().complete(messages, tools, max_tokens)
            return self._execution.complete(messages, tools, max_tokens)

    return CompareFakeModel


if __name__ == "__main__":
    app()
