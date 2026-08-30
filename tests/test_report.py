"""Iteration report tests (SPEC.md §17): block shape + provenance chain."""

from pathlib import Path

from experiments.evolution import EvolutionResult, IterationRecord, ProvenanceLink
from experiments.report import write_iteration_report


def _result() -> EvolutionResult:
    return EvolutionResult(
        workflow="onboarding",
        seed=7,
        iterations=[
            IterationRecord(
                iteration=1,
                runs_created=5,
                runs_succeeded=3,
                new_record_ids=["inventory-check-before-assignment"],
                candidate_id="candidate-01",
                decision="accepted",
                eval={
                    "from_version": "1",
                    "to_version": "2",
                    "previous_score": 0.7,
                    "candidate_score": 0.8,
                    "regressions": 0,
                    "score_vector": {"tool_calls": 2.0},
                },
                lines_modified=3,
            ),
            IterationRecord(
                iteration=2,
                runs_created=5,
                runs_succeeded=5,
                new_record_ids=[],
            ),
        ],
        provenance=[
            ProvenanceLink(
                iteration=1,
                candidate_id="candidate-01",
                record_ids=["inventory-check-before-assignment"],
                failure_run_ids=["run_001", "run_004"],
                previous_score=0.7,
                candidate_score=0.8,
                decision="accepted",
            )
        ],
    )


def test_report_contains_spec17_blocks_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "iteration-report.md"
    write_iteration_report(_result(), str(path))

    text = path.read_text(encoding="utf-8")
    assert "## Iteration 1" in text
    assert "## Iteration 2" in text
    assert "Training: 3 / 5 success" in text
    assert "+ inventory-check-before-assignment" in text
    assert "Candidate candidate-01:" in text
    assert "3 lines modified" in text
    assert "Validation:" in text
    assert "v1 70.0% -> v2 80.0%" in text
    assert "Regressions:" in text
    assert "Decision:\nACCEPT" in text
    # Iteration without a candidate (loop stopped early).
    assert "no patch proposed" in text
    # Provenance chain: failures -> evidence -> patch -> eval -> decision.
    assert "## Provenance chain" in text
    assert "run_001, run_004" in text
    assert "patch candidate-01" in text
    assert "0.70 -> 0.80" in text
    assert "ACCEPTED" in text
    # Footer: the held-out split stayed untouched.
    assert "held-out test set was not used" in text


def test_report_deterministic_given_same_result(tmp_path: Path) -> None:
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    write_iteration_report(_result(), str(first))
    write_iteration_report(_result(), str(second))
    assert first.read_bytes() == second.read_bytes()
