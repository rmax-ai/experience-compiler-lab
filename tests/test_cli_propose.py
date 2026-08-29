"""CLI propose smoke test: candidate artifacts written, deployed skill untouched."""

import json
from datetime import date
from pathlib import Path

import yaml
from typer.testing import CliRunner

import cli
from agent.adapter import FakeModel
from experiments.proposal_store import ProposalStore
from knowledge.schema import Claim, ClaimType, Evidence, KnowledgeRecord, Scope, Statistics
from knowledge.store import KnowledgeStore
from traces.store import TraceStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills" / "onboarding" / "SKILL.md"
runner = CliRunner()

_VALID_PAYLOAD = {
    "reasoning": (
        "The active knowledge record look-up-the-employee-record-before-granting-access "
        "supports verifying identity before granting access; assign hardware only when "
        "inventory confirms availability."
    ),
    "patch": "@@ Procedure\n"
    "- 4. Assign available hardware.\n"
    "+ 4. Assign hardware only if inventory confirms availability.\n",
}
_USAGE = {"input_tokens": 120, "output_tokens": 64}


def _record() -> KnowledgeRecord:
    return KnowledgeRecord(
        id="look-up-the-employee-record-before-granting-access",
        claim=Claim(
            type=ClaimType.procedure,
            text="Look up the employee record before granting access",
        ),
        scope=Scope(workflows=["onboarding"]),
        evidence=Evidence(supporting_runs=["run_001"], counterexamples=[]),
        statistics=Statistics(support=4, failures=0),
        confidence=0.83,
        status="active",
        first_seen=date(2026, 8, 1),
        last_updated=date(2026, 8, 1),
    )


def _proposer_factory() -> FakeModel:
    response = {"role": "assistant", "content": json.dumps(_VALID_PAYLOAD)}
    return FakeModel(
        scripts=[(response, dict(_USAGE)) for _ in range(2)],
        model="fake",
        temperature=0.0,
    )


def test_cli_propose_writes_candidate_and_never_touches_skill(
    tmp_path: Path, monkeypatch
) -> None:
    knowledge = KnowledgeStore(tmp_path)
    knowledge.upsert(_record())
    traces = TraceStore(tmp_path)
    store = ProposalStore(tmp_path)

    monkeypatch.setattr(cli, "KnowledgeStore", lambda: knowledge)
    monkeypatch.setattr(cli, "TraceStore", lambda: traces)
    monkeypatch.setattr(cli, "ProposalStore", lambda: store)
    monkeypatch.setattr(cli, "_proposer_model_factory", lambda model_name: _proposer_factory)

    skill_before = SKILL_PATH.read_bytes()

    result = runner.invoke(
        cli.app, ["propose", "onboarding", "--model", "fake", "--traces", "0"]
    )

    assert result.exit_code == 0, result.output
    assert "candidate: candidate-01" in result.output
    assert "patch_lines: 3" in result.output
    assert "look-up-the-employee-record-before-granting-access" in result.output
    assert "cost_usd:" in result.output

    # Candidate artifacts live under results/candidates/<id>/ only.
    candidate_dir = tmp_path / "results" / "candidates" / "candidate-01"
    assert (candidate_dir / "patch.md").exists()
    assert (candidate_dir / "reasoning.md").exists()
    assert (candidate_dir / "record.yaml").exists()

    record = yaml.safe_load((candidate_dir / "record.yaml").read_text(encoding="utf-8"))
    assert record["evidence_refs"] == ["look-up-the-employee-record-before-granting-access"]
    assert record["skill"] == "onboarding"
    assert record["decision"] == "pending"
    assert record["format_version"] == 1

    # Core isolation guarantee: the deployed skill is byte-for-byte unchanged.
    assert SKILL_PATH.read_bytes() == skill_before
    # And nothing was written under the real results/ dir.
    assert not (REPO_ROOT / "results" / "candidates" / "candidate-01").exists()


def test_cli_propose_no_patch_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    knowledge = KnowledgeStore(tmp_path)
    traces = TraceStore(tmp_path)
    store = ProposalStore(tmp_path)

    monkeypatch.setattr(cli, "KnowledgeStore", lambda: knowledge)
    monkeypatch.setattr(cli, "TraceStore", lambda: traces)
    monkeypatch.setattr(cli, "ProposalStore", lambda: store)

    def factory() -> FakeModel:
        response = {
            "role": "assistant",
            "content": json.dumps({"reasoning": "nothing worth changing", "patch": ""}),
        }
        return FakeModel(
            scripts=[(response, dict(_USAGE)) for _ in range(2)],
            model="fake",
            temperature=0.0,
        )

    monkeypatch.setattr(cli, "_proposer_model_factory", lambda model_name: factory)

    result = runner.invoke(
        cli.app, ["propose", "onboarding", "--model", "fake", "--traces", "0"]
    )
    assert result.exit_code == 0, result.output
    assert "no patch proposed" in result.output
    candidates_dir = tmp_path / "results" / "candidates"
    if candidates_dir.exists():
        assert list(candidates_dir.iterdir()) == []
