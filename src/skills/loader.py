"""Minimal skill loader (M3 extends this with the patch machinery).

Reads the deployed skill markdown and its PURPOSE.yaml provenance version.
Skills are the only mutable, reversible store — every change goes through a
patch with PURPOSE.yaml provenance (AGENTS.md §2).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"


def load_skill(workflow: str) -> str:
    """Return the raw markdown of ``skills/<workflow>/SKILL.md``.

    Raises ``FileNotFoundError`` naming the expected path when absent.
    """
    path = SKILLS_DIR / workflow / "SKILL.md"
    if not path.exists():
        raise FileNotFoundError(f"skill not found: {path}")
    return path.read_text(encoding="utf-8")


def get_skill_version(workflow: str) -> str:
    """Return the ``version`` field from ``skills/<workflow>/PURPOSE.yaml``."""
    path = SKILLS_DIR / workflow / "PURPOSE.yaml"
    if not path.exists():
        raise FileNotFoundError(f"skill metadata not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    version = data.get("version")
    return str(version) if version is not None else "unknown"
