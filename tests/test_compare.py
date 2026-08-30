"""M5 compare harness tests, including held-out persistence behaviour."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from cli import _compare_model_factory
from experiments import compare as compare_module
from experiments import evolution, promote
from experiments.compare import compare, write_compare_report
from experiments.runner import run_tasks
from skills import loader
from traces.schema import load_scenarios
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _configure_tmp_repo(monkeypatch, root: Path) -> None:  # noqa: ANN001
    shutil.copytree(REPO_ROOT / "datasets", root / "datasets")
    shutil.copytree(REPO_ROOT / "skills", root / "skills")
    skills_dir = root / "skills"
    monkeypatch.setattr(compare_module, "REPO_ROOT", root)
    monkeypatch.setattr(compare_module, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(evolution, "REPO_ROOT", root)
    monkeypatch.setattr(evolution, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(promote, "REPO_ROOT", root)
    monkeypatch.setattr(promote, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(loader, "SKILLS_DIR", skills_dir)


def _run_all(monkeypatch, root: Path):  # noqa: ANN001, ANN201
    _configure_tmp_repo(monkeypatch, root)
    return compare(
        ["baseline", "trace2skill", "memory", "compiler"],
        "onboarding",
        iterations=1,
        model_factory=_compare_model_factory("fake"),
        seed=42,
        dev_limit=2,
    )


def test_compare_end_to_end_and_report(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    results = _run_all(monkeypatch, tmp_path / "first")
    assert [result.config for result in results] == [
        "baseline",
        "trace2skill",
        "memory",
        "compiler",
    ]
    assert len({result.heldout_total for result in results}) == 1
    assert results[0].records_created == 0
    assert results[0].decisions == []
    assert results[2].records_created > 0
    assert results[2].skill_version == 1
    assert results[3].decisions

    md_path, csv_path = write_compare_report(results, 42)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert "## Hypotheses" in md_path.read_text(encoding="utf-8")


def test_compare_deterministic_rates_and_decisions(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    first = _run_all(monkeypatch, tmp_path / "first")
    second = _run_all(monkeypatch, tmp_path / "second")
    assert [(result.heldout_success_rate, result.decisions) for result in first] == [
        (result.heldout_success_rate, result.decisions) for result in second
    ]


def test_run_tasks_persist_false_does_not_append(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    scenario = load_scenarios(str(REPO_ROOT / "datasets" / "train.jsonl"))[:1]
    run_tasks(
        scenario,
        "onboarding",
        _compare_model_factory("fake"),
        seed=42,
        experiment_id="test-persist-false",
        store=store,
        persist=False,
    )
    assert store.list_runs() == []
