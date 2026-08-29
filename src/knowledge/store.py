"""Append-only knowledge store (AGENTS.md §2, SPEC.md §6).

``knowledge/patterns/<id>.yaml`` is the source of truth: records are created
or merged, never deleted. Merging only ever extends evidence (union of run
ids, sorted, deduped) and bumps ``last_updated``; ``first_seen`` and any
supersede links are preserved. ``knowledge/index.yaml`` is derived state —
a byte-deterministic summary rebuilt from the record files via
:meth:`KnowledgeStore.regenerate_index`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from knowledge.schema import Evidence, KnowledgeRecord, Statistics, from_yaml, to_yaml


class KnowledgeStore:
    """Rooted at ``knowledge/`` with ``patterns/<id>.yaml`` plus ``index.yaml``."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Root the store at ``base_dir`` (default: the repository root)."""
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        self.base_dir = Path(base_dir)
        self.patterns_dir = self.base_dir / "knowledge" / "patterns"
        self.index_path = self.base_dir / "knowledge" / "index.yaml"
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

    # -- path helpers ---------------------------------------------------------

    def record_path(self, record_id: str) -> Path:
        """Path of a record's YAML file (no existence check)."""
        self._validate_id(record_id)
        return self.patterns_dir / f"{record_id}.yaml"

    @staticmethod
    def _validate_id(record_id: str) -> None:
        if (
            not record_id
            or "/" in record_id
            or "\\" in record_id
            or ".." in record_id
        ):
            raise ValueError(f"invalid record id: {record_id!r}")

    # -- writes ---------------------------------------------------------------

    def upsert(self, record: KnowledgeRecord) -> KnowledgeRecord:
        """Write a new record, or merge evidence into the existing one.

        Merging extends ``supporting_runs``/``counterexamples`` (union,
        sorted, deduped — evidence is never removed), recomputes statistics
        and confidence, bumps ``last_updated`` to today, and keeps
        ``first_seen`` plus any supersede links and a ``superseded`` status.
        """
        path = self.record_path(record.id)
        if not path.exists():
            path.write_text(to_yaml(record) + "\n", encoding="utf-8")
            return record

        stored = from_yaml(path.read_text(encoding="utf-8"))
        supporting = sorted(set(stored.evidence.supporting_runs) | set(record.evidence.supporting_runs))
        counterexamples = sorted(
            set(stored.evidence.counterexamples) | set(record.evidence.counterexamples)
        )
        support = len(supporting)
        failures = len(counterexamples)
        merged = KnowledgeRecord(
            id=stored.id,
            claim=stored.claim,
            scope=stored.scope,
            evidence=Evidence(supporting_runs=supporting, counterexamples=counterexamples),
            statistics=Statistics(support=support, failures=failures),
            confidence=(support + 1) / (support + failures + 2),
            status="superseded" if stored.status == "superseded" else record.status,
            first_seen=stored.first_seen,
            last_updated=date.today(),
            supersedes=stored.supersedes if stored.supersedes is not None else record.supersedes,
            superseded_by=(
                stored.superseded_by if stored.superseded_by is not None else record.superseded_by
            ),
            format_version=stored.format_version,
        )
        path.write_text(to_yaml(merged) + "\n", encoding="utf-8")
        return merged

    def supersede(self, old_id: str, new_id: str) -> None:
        """Retire ``old_id`` in favor of ``new_id``; nothing is ever deleted.

        The old record becomes ``status: superseded`` with
        ``superseded_by: [new_id]``; the new record gains ``supersedes:
        [old_id]``. The index is regenerated so it reflects the new statuses.
        """
        old = self.get(old_id)
        new = self.get(new_id)
        old.status = "superseded"
        old.superseded_by = sorted(set(old.superseded_by or []) | {new_id})
        old.last_updated = date.today()
        self.record_path(old.id).write_text(to_yaml(old) + "\n", encoding="utf-8")

        new.supersedes = sorted(set(new.supersedes or []) | {old_id})
        new.last_updated = date.today()
        self.record_path(new.id).write_text(to_yaml(new) + "\n", encoding="utf-8")

        self.regenerate_index()

    # -- reads ----------------------------------------------------------------

    def get(self, record_id: str) -> KnowledgeRecord:
        """Read one record from its YAML file (the source of truth)."""
        path = self.record_path(record_id)
        with path.open(encoding="utf-8") as handle:
            return from_yaml(handle.read())

    def all_records(self) -> list[KnowledgeRecord]:
        """All records from the pattern files, sorted by id."""
        records = [
            from_yaml(path.read_text(encoding="utf-8"))
            for path in sorted(self.patterns_dir.glob("*.yaml"))
        ]
        return sorted(records, key=lambda record: record.id)

    # -- maintenance ------------------------------------------------------------

    def regenerate_index(self) -> None:
        """Write ``knowledge/index.yaml`` (derived state, byte-deterministic).

        The index carries no timestamps and sorts keys (sort_keys=True), so
        identical records always produce an identical file.
        """
        records = self.all_records()
        index = {
            "format_version": 1,
            "record_count": len(records),
            "records": [
                {
                    "id": record.id,
                    "claim_type": record.claim.type.value,
                    "status": record.status,
                    "support": record.statistics.support,
                    "failures": record.statistics.failures,
                    "confidence": record.confidence,
                    "workflows": list(record.scope.workflows),
                }
                for record in records
            ],
        }
        self.index_path.write_text(
            yaml.safe_dump(index, sort_keys=True) + "\n", encoding="utf-8"
        )
