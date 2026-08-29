"""Immutable trace store (docs/data-formats.md §2, AGENTS.md §2).

``experience/runs/<run_id>.jsonl`` is the source of truth (append-only,
exclusive create); ``experience/index.db`` is a disposable SQLite index that
can always be rebuilt from the JSONL files. WAL is off, one connection per
operation, thread-unsafe is fine (the runner is single-threaded).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from traces.schema import Trace, from_jsonl

_RUN_FILE = re.compile(r"^run_(\d+)\.jsonl$")

_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT,
    model TEXT,
    skill_version TEXT,
    success INTEGER,
    tool_calls INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    estimated_cost_usd REAL,
    timestamp TEXT
)
"""

_INDEX_COLUMNS = (
    "run_id",
    "task_id",
    "model",
    "skill_version",
    "success",
    "tool_calls",
    "tokens_in",
    "tokens_out",
    "estimated_cost_usd",
    "timestamp",
)


class TraceStore:
    """Append-only JSONL run files plus a rebuildable SQLite index."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Root the store at ``base_dir`` (default: the repository root)."""
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "experience" / "runs"
        self.index_path = self.base_dir / "experience" / "index.db"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # -- path helpers -------------------------------------------------------

    def run_path(self, run_id: str) -> Path:
        """Path of a run's JSONL file (no existence check)."""
        return self.runs_dir / f"{run_id}.jsonl"

    # -- writes -------------------------------------------------------------

    def append(self, trace: Trace) -> Path:
        """Write the run JSONL (exclusive create) then index it.

        Raises ``FileExistsError`` if ``run_id`` already exists — the store is
        immutable and a run file is never rewritten.
        """
        path = self.run_path(trace.run_id)
        with path.open("x", encoding="utf-8") as handle:  # exclusive create
            handle.write(trace.model_dump_json())
            handle.write("\n")
        self._insert_row(trace)
        return path

    def _insert_row(self, trace: Trace) -> None:
        row = self._row(trace)
        placeholders = ", ".join("?" for _ in _INDEX_COLUMNS)
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO runs ({', '.join(_INDEX_COLUMNS)}) VALUES ({placeholders})",
                tuple(row[col] for col in _INDEX_COLUMNS),
            )
            conn.commit()
        finally:
            conn.close()

    # -- reads --------------------------------------------------------------

    def get(self, run_id: str) -> Trace:
        """Read a run from its JSONL file (the source of truth)."""
        path = self.run_path(run_id)
        with path.open(encoding="utf-8") as handle:
            return from_jsonl(handle.read())

    def list_runs(
        self,
        task_id: str | None = None,
        model: str | None = None,
        success: bool | None = None,
    ) -> list[dict]:
        """Index rows, optionally filtered, ordered by run_id."""
        clauses: list[str] = []
        params: list[object] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if success is not None:
            clauses.append("success = ?")
            params.append(int(success))

        query = f"SELECT {', '.join(_INDEX_COLUMNS)} FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY run_id"

        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()
        return [dict(zip(_INDEX_COLUMNS, row, strict=True)) for row in rows]

    # -- maintenance ----------------------------------------------------------

    def rebuild_index(self) -> None:
        """Drop and rebuild the SQLite index from the JSONL run files.

        The JSONL files are the source of truth; the index is derived state.
        """
        conn = self._connect()
        try:
            conn.execute("DROP TABLE IF EXISTS runs")
            conn.execute(_INDEX_SCHEMA)
            for path in sorted(self.runs_dir.glob("run_*.jsonl")):
                trace = from_jsonl(path.read_text(encoding="utf-8"))
                row = self._row(trace)
                placeholders = ", ".join("?" for _ in _INDEX_COLUMNS)
                conn.execute(
                    f"INSERT INTO runs ({', '.join(_INDEX_COLUMNS)}) VALUES ({placeholders})",
                    tuple(row[col] for col in _INDEX_COLUMNS),
                )
            conn.commit()
        finally:
            conn.close()

    def next_run_id(self) -> str:
        """Deterministic run id: ``run_<max existing index + 1>`` (zero-padded).

        Scans the JSONL directory (source of truth); an empty store yields
        ``run_001``.
        """
        max_index = 0
        for path in self.runs_dir.glob("run_*.jsonl"):
            match = _RUN_FILE.fullmatch(path.name)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
        return f"run_{max_index + 1:03d}"

    # -- internals -------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.index_path)
        conn.execute("PRAGMA journal_mode=DELETE")  # WAL off
        conn.execute(_INDEX_SCHEMA)
        return conn

    @staticmethod
    def _row(trace: Trace) -> dict[str, object]:
        """Derive an index row entirely from the trace (JSONL is the source of
        truth, so ``rebuild_index`` reproduces byte-identical rows)."""
        return {
            "run_id": trace.run_id,
            "task_id": trace.task_id,
            "model": trace.model,
            "skill_version": trace.skill_version,
            "success": int(trace.outcome.success),
            "tool_calls": trace.metrics.tool_calls,
            "tokens_in": trace.metrics.tokens_in,
            "tokens_out": trace.metrics.tokens_out,
            "estimated_cost_usd": trace.metrics.estimated_cost_usd,
            "timestamp": TraceStore._run_timestamp(trace),
        }

    @staticmethod
    def _run_timestamp(trace: Trace) -> str:
        """ISO timestamp of the last action, or '' for action-less traces."""
        if trace.actions:
            return trace.actions[-1].timestamp.isoformat()
        return ""
