"""ProposalStore tests: deterministic ids, artifact roundtrip, history shape."""

from pathlib import Path

import pytest
import yaml

from experiments.proposal_store import ProposalStore
from skills.proposer import Proposal

_PATCH = "@@ Procedure\n- 4. Assign available hardware.\n+ 4. Assign hardware only if inventory confirms availability.\n"


def _proposal() -> Proposal:
    return Proposal(
        reasoning="The record inventory-check-before-assignment supports checking inventory first.",
        patch=_PATCH,
        cited_records=["inventory-check-before-assignment"],
    )


def test_next_candidate_id_deterministic_on_empty_dir(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    assert store.next_candidate_id() == "candidate-01"
    assert store.next_candidate_id() == "candidate-01"


def test_next_candidate_id_increments_after_save(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    store.save_candidate(
        candidate_id="candidate-01",
        proposal=_proposal(),
        workflow="onboarding",
        from_version="1",
        to_version="2",
        proposed_model="fake",
    )
    assert store.next_candidate_id() == "candidate-02"


def test_save_load_roundtrip_all_three_artifacts(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    store.save_candidate(
        candidate_id="candidate-01",
        proposal=_proposal(),
        workflow="onboarding",
        from_version="1",
        to_version="2",
        proposed_model="fake",
    )

    directory = tmp_path / "results" / "candidates" / "candidate-01"
    assert (directory / "patch.md").read_text(encoding="utf-8").rstrip("\n") == _PATCH.rstrip("\n")
    assert "checking inventory first" in (directory / "reasoning.md").read_text(encoding="utf-8")

    candidate = store.load_candidate("candidate-01")
    assert candidate["candidate_id"] == "candidate-01"
    assert candidate["patch"].rstrip("\n") == _PATCH.rstrip("\n")
    assert candidate["reasoning"].startswith("The record inventory-check-before-assignment")

    record = candidate["record"]
    assert record["candidate_id"] == "candidate-01"
    assert record["skill"] == "onboarding"
    assert record["from_version"] == "1"
    assert record["to_version"] == "2"
    assert record["diff_file"] == "results/candidates/candidate-01/patch.md"
    assert record["evidence_refs"] == ["inventory-check-before-assignment"]
    assert record["evaluation"] == {"previous_score": None, "candidate_score": None}
    assert record["decision"] == "pending"
    assert record["decided_at"] is None
    assert record["format_version"] == 1
    assert record["proposed_by"] == {"model": "fake"}


def test_list_candidates_sorted_by_id(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    for candidate_id in ("candidate-02", "candidate-01", "candidate-10"):
        store.save_candidate(
            candidate_id=candidate_id,
            proposal=_proposal(),
            workflow="onboarding",
            from_version="1",
            to_version="2",
            proposed_model="fake",
        )
    assert store.list_candidates() == ["candidate-01", "candidate-02", "candidate-10"]


def test_load_history_shape_used_by_proposer(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    store.save_candidate(
        candidate_id="candidate-01",
        proposal=_proposal(),
        workflow="onboarding",
        from_version="1",
        to_version="2",
        proposed_model="fake",
    )
    store.save_candidate(
        candidate_id="candidate-02",
        proposal=Proposal(reasoning="second", patch="@@ Procedure\n- 2. Retrieve onboarding requirements.\n", cited_records=[]),
        workflow="onboarding",
        from_version="1",
        to_version="2",
        proposed_model="fake",
    )
    # Simulate a decided candidate.
    record_path = tmp_path / "results" / "candidates" / "candidate-02" / "record.yaml"
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    record["decision"] = "accepted"
    record_path.write_text(yaml.safe_dump(record, sort_keys=True) + "\n", encoding="utf-8")

    history = store.load_history()
    assert history == [
        {
            "candidate_id": "candidate-01",
            "skill": "onboarding",
            "decision": "pending",
            "patch_headline": "@@ Procedure",
        },
        {
            "candidate_id": "candidate-02",
            "skill": "onboarding",
            "decision": "accepted",
            "patch_headline": "@@ Procedure",
        },
    ]


def test_load_candidate_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProposalStore(tmp_path).load_candidate("candidate-99")
