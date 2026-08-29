"""Candidate/proposal store (docs/data-formats.md §6, AGENTS.md §2).

Rooted at ``results/candidates/<id>/`` with three artifacts per candidate:
``patch.md`` (the diff text), ``reasoning.md`` (why), and ``record.yaml``
(the structured proposal record: provenance, evidence refs, evaluation
slots, decision). This is the ONLY place the proposer writes — deployed
skills under ``skills/`` are touched exclusively by promotion (M4), never by
this module (the M3 isolation rule).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from skills.proposer import Proposal

_CANDIDATE_RE = re.compile(r"^candidate-(\d+)$")


class ProposalStore:
    """Candidate storage rooted at ``results/candidates/``."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Root the store at ``base_dir`` (default: the repository root)."""
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        self.base_dir = Path(base_dir)
        self.candidates_dir = self.base_dir / "results" / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    # -- ids ------------------------------------------------------------------

    def next_candidate_id(self) -> str:
        """Deterministic id: ``candidate-<max existing number + 1>`` zero-padded.

        An empty store yields ``candidate-01``.
        """
        max_index = 0
        for path in self.candidates_dir.iterdir():
            if path.is_dir():
                match = _CANDIDATE_RE.fullmatch(path.name)
                if match is not None:
                    max_index = max(max_index, int(match.group(1)))
        return f"candidate-{max_index + 1:02d}"

    # -- writes ---------------------------------------------------------------

    def save_candidate(
        self,
        candidate_id: str,
        proposal: Proposal,
        workflow: str,
        from_version: str,
        to_version: str,
        proposed_model: str,
    ) -> Path:
        """Write the three candidate artifacts under ``results/candidates/<id>/``.

        ``record.yaml`` follows docs/data-formats.md §6: provenance, evidence
        refs (the proposal's cited records), null evaluation slots, decision
        ``pending``. Re-saving an existing id overwrites its artifacts.
        """
        directory = self.candidates_dir / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "patch.md").write_text(
            _ensure_trailing_newline(proposal.patch), encoding="utf-8"
        )
        (directory / "reasoning.md").write_text(
            _ensure_trailing_newline(proposal.reasoning), encoding="utf-8"
        )
        record = {
            "candidate_id": candidate_id,
            "skill": workflow,
            "from_version": from_version,
            "to_version": to_version,
            "diff_file": f"results/candidates/{candidate_id}/patch.md",
            "evidence_refs": list(proposal.cited_records),
            "evaluation": {"previous_score": None, "candidate_score": None},
            "decision": "pending",
            "decided_at": None,
            "proposed_by": {"model": proposed_model},
            "format_version": 1,
        }
        (directory / "record.yaml").write_text(
            yaml.safe_dump(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        return directory

    # -- reads ----------------------------------------------------------------

    def load_candidate(self, candidate_id: str) -> dict[str, object]:
        """Load all three artifacts for one candidate.

        Returns ``{"candidate_id", "patch", "reasoning", "record"}`` where
        ``record`` is the parsed ``record.yaml`` dict. Raises
        ``FileNotFoundError`` when the candidate directory is missing.
        """
        directory = self.candidates_dir / candidate_id
        return {
            "candidate_id": candidate_id,
            "patch": (directory / "patch.md").read_text(encoding="utf-8"),
            "reasoning": (directory / "reasoning.md").read_text(encoding="utf-8"),
            "record": yaml.safe_load(
                (directory / "record.yaml").read_text(encoding="utf-8")
            )
            or {},
        }

    def list_candidates(self) -> list[str]:
        """Candidate ids sorted by id (zero-padded, so lexicographic == numeric)."""
        candidate_ids: list[str] = []
        for path in self.candidates_dir.iterdir():
            if path.is_dir() and _CANDIDATE_RE.fullmatch(path.name) is not None:
                candidate_ids.append(path.name)
        return sorted(candidate_ids)

    def load_history(self) -> list[dict[str, object]]:
        """Compact proposal history for the proposer (docs/data-formats.md §6).

        One entry per candidate: ``candidate_id``, ``skill``, ``decision``
        (null becomes ``"pending"``) and ``patch_headline`` (the first
        non-blank patch line). Sorted by candidate id.
        """
        history: list[dict[str, object]] = []
        for candidate_id in self.list_candidates():
            record = yaml.safe_load(
                (self.candidates_dir / candidate_id / "record.yaml").read_text(
                    encoding="utf-8"
                )
            ) or {}
            decision = record.get("decision")
            patch = (self.candidates_dir / candidate_id / "patch.md").read_text(
                encoding="utf-8"
            )
            history.append(
                {
                    "candidate_id": candidate_id,
                    "skill": record.get("skill"),
                    "decision": decision if decision is not None else "pending",
                    "patch_headline": _patch_headline(patch),
                }
            )
        return sorted(history, key=lambda entry: str(entry["candidate_id"]))


def _patch_headline(patch_text: str) -> str:
    """First non-blank line of a patch ('' for an empty patch)."""
    for line in patch_text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
