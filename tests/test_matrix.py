"""M6 transfer-matrix tests (SPEC.md §13/§16) plus the held-out split guard.

Isolation mirrors test_compare.py: each run copies datasets/ and skills/ into
a tmp tree and repoints the module-level REPO_ROOT / SKILLS_DIR of every
module that resolves them at call time. The fake model is the deterministic
role-routing ``_matrix_model_factory`` from the CLI.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from cli import _matrix_model_factory
from experiments import evolution, promote
from experiments import matrix as matrix_module
from experiments.matrix import (
    run_transfer_matrix,
    write_transfer_matrix_csv,
    write_transfer_matrix_report,
)
from skills import loader

REPO_ROOT = Path(__file__).resolve().parents[1]

CSV_COLUMNS = [
    "skill_source",
    "executor_model",
    "heldout_success_rate",
    "decisions",
    "records_created",
    "skill_version",
    "cost_usd",
    "seed",
]


def _configure_tmp_repo(monkeypatch, root: Path) -> None:  # noqa: ANN001
    shutil.copytree(REPO_ROOT / "datasets", root / "datasets")
    shutil.copytree(REPO_ROOT / "skills", root / "skills")
    skills_dir = root / "skills"
    monkeypatch.setattr(matrix_module, "REPO_ROOT", root)
    monkeypatch.setattr(matrix_module, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(evolution, "REPO_ROOT", root)
    monkeypatch.setattr(evolution, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(promote, "REPO_ROOT", root)
    monkeypatch.setattr(promote, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(loader, "SKILLS_DIR", skills_dir)


def _skill_bytes(root: Path) -> tuple[bytes, bytes]:
    workflow = root / "skills" / "onboarding"
    return (
        (workflow / "SKILL.md").read_bytes(),
        (workflow / "PURPOSE.yaml").read_bytes(),
    )


def _run_matrix(monkeypatch, root: Path, seed: int = 42):  # noqa: ANN001, ANN202
    _configure_tmp_repo(monkeypatch, root)
    return run_transfer_matrix(
        ["fake"], "onboarding", 1, _matrix_model_factory, seed, dev_limit=2
    )


def test_matrix_end_to_end_and_restoration(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    root = tmp_path / "run"
    _configure_tmp_repo(monkeypatch, root)
    before = _skill_bytes(root)

    result = run_transfer_matrix(
        ["fake"], "onboarding", 1, _matrix_model_factory, 42, dev_limit=2
    )

    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell.skill_source == "fake"
    assert cell.executor_model == "fake"
    assert 0.0 <= cell.heldout_success_rate <= 1.0
    assert cell.seed == 42

    # skills/ restored byte-identically after the full run.
    assert _skill_bytes(root) == before

    # Variant stashed with skill text and provenance copy.
    stash = root / "results" / "transfer-skills" / "fake"
    assert (stash / "SKILL.md").exists()
    assert (stash / "PURPOSE.yaml").exists()
    assert (stash / "PURPOSE.yaml").read_bytes() == (
        root / "skills" / "onboarding" / "PURPOSE.yaml"
    ).read_bytes()

    csv_path = write_transfer_matrix_csv(result, root / "results" / "transfer-matrix.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["skill_source"] == "fake"
    assert rows[0]["executor_model"] == "fake"


def test_matrix_deterministic_for_same_seed(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    first = _run_matrix(monkeypatch, tmp_path / "first")
    second = _run_matrix(monkeypatch, tmp_path / "second")
    assert [(c.heldout_success_rate, c.decisions, c.cost_usd) for c in first.cells] == [
        (c.heldout_success_rate, c.decisions, c.cost_usd) for c in second.cells
    ]


def test_matrix_csv_columns_and_report_sections(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    root = tmp_path / "run"
    result = _run_matrix(monkeypatch, root)

    csv_path = write_transfer_matrix_csv(result, root / "results" / "transfer-matrix.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == CSV_COLUMNS

    md_path = write_transfer_matrix_report(
        result, root / "results" / "transfer-matrix.md", iterations=1
    )
    report = md_path.read_text(encoding="utf-8")
    assert "## Transfer matrix" in report
    assert "| skill source \\ executor | fake |" in report
    assert "## Per-cell cost" in report
    assert "Grand total cost:" in report
    assert "## Reproducibility" in report
    assert "- seed: 42" in report
    assert "- models: fake" in report


def test_held_out_split_only_referenced_in_allowed_modules() -> None:
    """SPEC.md §10: test.jsonl is M5/M6-only (compare, matrix, CLI wiring)."""
    allowed = {
        REPO_ROOT / "src" / "experiments" / "compare.py",
        REPO_ROOT / "src" / "experiments" / "matrix.py",
        REPO_ROOT / "src" / "cli.py",
    }
    offenders = [
        path
        for path in sorted((REPO_ROOT / "src").rglob("*.py"))
        if "test" + ".jsonl" in path.read_text(encoding="utf-8") and path not in allowed
    ]
    assert offenders == [], f"held-out split referenced outside allowed modules: {offenders}"


def test_matrix_cost_evidence(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    result = _run_matrix(monkeypatch, tmp_path / "run")
    assert all(cell.cost_usd >= 0 for cell in result.cells)
    # The fake dev scripts carry non-zero token usage, so phase-1 training
    # runs alone push the single fake cell's cost above zero (AGENTS.md §9).
    assert result.cells[0].cost_usd > 0


def test_matrix_restores_skills_when_run_tasks_raises(  # noqa: ANN001
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "run"
    _configure_tmp_repo(monkeypatch, root)
    before = _skill_bytes(root)

    def raising_factory(model_name: str):  # noqa: ANN202
        def factory():  # noqa: ANN202
            raise RuntimeError(f"boom from {model_name}")

        return factory

    with pytest.raises(RuntimeError, match="boom"):
        run_transfer_matrix(["fake"], "onboarding", 1, raising_factory, 42, dev_limit=2)

    assert _skill_bytes(root) == before
