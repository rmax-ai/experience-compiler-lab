"""M6 cross-model transfer matrix (SPEC.md §13 and §16).

Train one skill variant per source model via the M4 evolution loop (phase 1),
then execute EVERY variant on EVERY model against the held-out test split
(phase 2). Only this module (besides experiments/compare.py) opens
``datasets/test.jsonl`` — the split stays held out: no training, evaluation,
or tuning decision ever reads it, and per-cell runs are the first time a
variant sees those scenarios (SPEC.md §10).

Invariants (AGENTS.md §2/§3):
- ``skills/<workflow>/`` is restored byte-identically after every phase and
  after the full run; variants live only under ``results/transfer-skills/``.
- This module never imports ``knowledge`` — the executor context stays clean
  and ``evolve`` is called as a black box.
- Per-cell cost evidence sums the source model's training runs (phase 1, from
  the trace store) plus the cell's held-out execution traces (phase 2).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from experiments.evolution import evolve
from experiments.promote import load_skill_version
from experiments.runner import ModelFactory, run_tasks
from skills.loader import SKILLS_DIR
from traces.schema import load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[2]

ModelFactoryFor = Callable[[str], ModelFactory]


@dataclass
class MatrixCell:
    """One (skill source, executor model) transfer measurement."""

    skill_source: str
    executor_model: str
    heldout_success_rate: float
    decisions: list[str] = field(default_factory=list)
    records_created: int = 0
    skill_version: int = 0
    cost_usd: float = 0.0
    seed: int = 0


@dataclass
class MatrixResult:
    """All cells of one matrix run, ordered by (skill_source, executor_model)."""

    cells: list[MatrixCell]
    seed: int


@dataclass
class _Variant:
    """Phase-1 training output for one source model."""

    skill_md: str
    skill_version: int
    decisions: list[str]
    records_created: int
    train_cost_usd: float


def run_transfer_matrix(
    models: list[str],
    workflow: str,
    iterations: int,
    model_factory_for: ModelFactoryFor,
    seed: int,
    dev_limit: int | None = None,
) -> MatrixResult:
    """Train one skill per source model, then run every skill on every model.

    Phase 1 restores the deployed skill, evolves a variant with the source
    model, and stashes it under ``results/transfer-skills/<source>/``. Phase 2
    swaps each stashed variant into ``skills/<workflow>/SKILL.md`` one cell at
    a time (the backup PURPOSE.yaml stays in place) and runs the held-out
    split with the executor model at cell seed
    ``seed + source_index * 100 + executor_index``. Both phases restore the
    deployed skill in a ``finally`` block, so a failure never leaves a variant
    behind.
    """
    if not models:
        raise ValueError("models must contain at least one model name")

    skill_dir = SKILLS_DIR / workflow
    skill_path = skill_dir / "SKILL.md"
    purpose_path = skill_dir / "PURPOSE.yaml"
    skill_backup = skill_path.read_bytes()
    purpose_backup = purpose_path.read_bytes()

    def restore() -> None:
        skill_path.write_bytes(skill_backup)
        purpose_path.write_bytes(purpose_backup)

    heldout = load_scenarios(str(REPO_ROOT / "datasets" / "test.jsonl"))
    store = TraceStore(REPO_ROOT)
    stash_root = REPO_ROOT / "results" / "transfer-skills"

    variants: dict[str, _Variant] = {}
    try:
        for source in models:
            restore()
            runs_before = {row["run_id"] for row in store.list_runs()}
            evolution = evolve(
                workflow, iterations, model_factory_for(source), seed, dev_limit or 0
            )
            train_cost = sum(
                float(row["estimated_cost_usd"])
                for row in store.list_runs()
                if row["run_id"] not in runs_before
            )
            stash_dir = stash_root / source
            stash_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(skill_path, stash_dir / "SKILL.md")
            shutil.copyfile(purpose_path, stash_dir / "PURPOSE.yaml")
            variants[source] = _Variant(
                skill_md=(stash_dir / "SKILL.md").read_text(encoding="utf-8"),
                skill_version=_version_as_int(load_skill_version(SKILLS_DIR, workflow)),
                decisions=[
                    record.decision
                    for record in evolution.iterations
                    if record.decision is not None
                ],
                records_created=len(
                    {
                        record_id
                        for record in evolution.iterations
                        for record_id in record.new_record_ids
                    }
                ),
                train_cost_usd=train_cost,
            )
    finally:
        restore()

    cells: list[MatrixCell] = []
    try:
        for source_index, source in enumerate(models):
            variant = variants[source]
            for executor_index, executor in enumerate(models):
                cell_seed = seed + source_index * 100 + executor_index
                restore()
                try:
                    skill_path.write_text(variant.skill_md, encoding="utf-8")
                    traces = run_tasks(
                        heldout,
                        workflow,
                        model_factory_for(executor),
                        cell_seed,
                        f"matrix-{source}-{executor}-{seed}",
                        store=TraceStore(REPO_ROOT / "results"),
                        persist=False,
                    )
                finally:
                    restore()
                total = len(traces)
                rate = (
                    round(sum(trace.outcome.success for trace in traces) / total, 4)
                    if total
                    else 0.0
                )
                exec_cost = sum(trace.metrics.estimated_cost_usd for trace in traces)
                cells.append(
                    MatrixCell(
                        skill_source=source,
                        executor_model=executor,
                        heldout_success_rate=rate,
                        decisions=list(variant.decisions),
                        records_created=variant.records_created,
                        skill_version=variant.skill_version,
                        cost_usd=variant.train_cost_usd + exec_cost,
                        seed=cell_seed,
                    )
                )
    finally:
        restore()

    return MatrixResult(cells=cells, seed=seed)


def write_transfer_matrix_csv(result: MatrixResult, path: str | Path) -> Path:
    """Write one row per cell (SPEC.md §16: ``results/transfer-matrix.csv``)."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "skill_source",
        "executor_model",
        "heldout_success_rate",
        "decisions",
        "records_created",
        "skill_version",
        "cost_usd",
        "seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cell in result.cells:
            writer.writerow(
                {
                    "skill_source": cell.skill_source,
                    "executor_model": cell.executor_model,
                    "heldout_success_rate": f"{cell.heldout_success_rate:.4f}",
                    "decisions": ",".join(cell.decisions),
                    "records_created": cell.records_created,
                    "skill_version": cell.skill_version,
                    "cost_usd": f"{cell.cost_usd:.6f}",
                    "seed": cell.seed,
                }
            )
    return csv_path


def write_transfer_matrix_report(
    result: MatrixResult, path: str | Path, iterations: int | None = None
) -> Path:
    """Write the SPEC §13 matrix table, per-cell costs, and reproducibility."""
    md_path = Path(path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    sources = list(dict.fromkeys(cell.skill_source for cell in result.cells))
    executors = list(dict.fromkeys(cell.executor_model for cell in result.cells))
    by_cell = {(cell.skill_source, cell.executor_model): cell for cell in result.cells}

    lines = [
        "# M6 cross-model transfer matrix",
        "",
        "## Transfer matrix",
        "",
        "Held-out success rate per cell (rows = skill source, columns = executor model).",
        "",
        "| skill source \\ executor | " + " | ".join(executors) + " |",
        "| " + " | ".join("---" for _ in range(len(executors) + 1)) + " |",
    ]
    for source in sources:
        row = [
            f"{by_cell[(source, executor)].heldout_success_rate:.2%}" for executor in executors
        ]
        lines.append("| " + source + " | " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Per-cell cost",
            "",
            "| skill_source | executor_model | cost_usd |",
            "| --- | --- | --- |",
        ]
    )
    for cell in result.cells:
        lines.append(
            f"| {cell.skill_source} | {cell.executor_model} | {cell.cost_usd:.6f} |"
        )
    grand_total = sum(cell.cost_usd for cell in result.cells)
    lines.extend(["", f"Grand total cost: {grand_total:.6f} USD", ""])

    lines.extend(
        [
            "## Reproducibility",
            "",
            f"- seed: {result.seed}",
            f"- iterations: {iterations if iterations is not None else 'unknown'}",
            f"- dataset git version: {_git_head()}",
            f"- models: {', '.join(sources)}",
            "",
            "Cell seed = seed + source_index * 100 + executor_index; per-cell costs",
            "include the source model's phase-1 training runs and the cell's",
            "held-out execution traces.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _git_head() -> str:
    """Full ``git rev-parse HEAD`` for the reproducibility block (SPEC.md §19)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
    if proc.returncode != 0:
        return "uncommitted"
    return proc.stdout.strip()


def _version_as_int(version: str) -> int:
    try:
        return int(version)
    except ValueError:
        return 0
