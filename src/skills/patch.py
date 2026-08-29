"""Deterministic skill patch grammar (docs/data-formats.md §5, SPEC.md §9).

Patches are restricted to ``@@ <section>`` headers plus ``- `` (removed) and
``+ `` (added) lines, with optional verbatim context anchors. Applying is
purely mechanical: a hunk's context/removed lines must match one contiguous
verbatim block inside the target section, or the patch is rejected with a
precise :class:`PatchError`. Ambiguity (the same block matching in several
places) is also rejected — the patch author must add context anchors to
disambiguate. Patches never partially apply: ``apply_patch`` builds a fresh
copy of the skill and returns it only on full success, so a failed hunk
leaves the skill untouched (AGENTS.md §2 — deployed skills change only via
applied patches with PURPOSE.yaml provenance).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

_HUNK_HEADER = re.compile(r"^@@\s+(.+?)\s*$")


class PatchError(Exception):
    """Raised for any patch that cannot be parsed or applied cleanly."""


class PatchHunk(BaseModel):
    """One ``@@ Section`` hunk.

    ``context`` holds verbatim anchor lines that stay in the file, ``removed``
    lines are matched verbatim and replaced by ``added`` lines. The match
    needle is ``context + removed`` (in patch order) and the replacement is
    ``context + added``.
    """

    model_config = ConfigDict(extra="forbid")

    section: str
    context: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)


def parse_patch(text: str, skill_md: str | None = None) -> list[PatchHunk]:
    """Parse patch text into hunks.

    Grammar: blank lines are ignored; ``@@ <Section>`` starts a hunk (the
    section name is matched case-insensitively, spaces trimmed, against the
    ``## <Section>`` headers of ``skill_md`` when it is provided); ``- `` and
    ``+ `` lines are removed/added content; any other non-empty line is a
    context anchor.

    Raises :class:`PatchError` on a hunk header naming a section that does
    not exist in the skill, junk/malformed lines, an empty hunk (no removed
    and no added lines), or an empty patch.
    """
    section_names = _section_names(skill_md) if skill_md is not None else None

    hunks: list[PatchHunk] = []
    current: PatchHunk | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if raw.startswith("@@"):
            match = _HUNK_HEADER.match(raw)
            if match is None:
                raise PatchError(f"malformed hunk header: {line!r}")
            section = match.group(1).strip()
            if not section:
                raise PatchError("hunk header names an empty section")
            if section_names is not None and _normalize_section(section) not in section_names:
                raise PatchError(f"patch names unknown section: {section!r}")
            current = PatchHunk(section=section)
            hunks.append(current)
            continue
        if current is None:
            raise PatchError(f"junk before first hunk header: {line!r}")
        if raw.startswith("- "):
            content = raw[2:]
            if not content.strip():
                raise PatchError(f"empty removed line: {line!r}")
            current.removed.append(content)
        elif raw.startswith("+ "):
            content = raw[2:]
            if not content.strip():
                raise PatchError(f"empty added line: {line!r}")
            current.added.append(content)
        elif raw.startswith("-") or raw.startswith("+"):
            raise PatchError(f"malformed hunk line (need '- ' or '+ '): {line!r}")
        elif raw.startswith("@"):
            raise PatchError(f"junk hunk line: {line!r}")
        else:
            current.context.append(raw)

    if not hunks:
        raise PatchError("empty patch")
    for hunk in hunks:
        if not hunk.removed and not hunk.added:
            raise PatchError(f"empty hunk for section {hunk.section!r}")
    return hunks


def apply_patch(skill_md: str, patch_text: str) -> str:
    """Apply ``patch_text`` to ``skill_md`` and return the resulting markdown.

    The skill is parsed into sections delimited by ``## `` lines; the
    preamble and section headers are preserved verbatim. Every hunk must
    apply cleanly to its section or :class:`PatchError` is raised and no
    result is returned (no partial application). Deterministic: identical
    inputs always produce identical outputs.
    """
    hunks = parse_patch(patch_text, skill_md)
    preamble, sections = _split_sections(skill_md)

    by_name: dict[str, list[int]] = {}
    for index, (header, _body) in enumerate(sections):
        by_name.setdefault(_normalize_section(header[3:]), []).append(index)

    working = [list(body) for _header, body in sections]
    for hunk in hunks:
        indices = by_name.get(_normalize_section(hunk.section), [])
        if not indices:
            raise PatchError(f"patch does not apply: no section {hunk.section!r}")
        if len(indices) > 1:
            raise PatchError(f"patch does not apply: ambiguous section {hunk.section!r}")
        _apply_hunk(working[indices[0]], hunk)

    out_lines: list[str] = list(preamble)
    for (header, _body), body in zip(sections, working, strict=True):
        out_lines.append(header)
        out_lines.extend(body)
    result = "\n".join(out_lines)
    if skill_md.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def verify_applies(skill_md: str, patch_text: str) -> bool:
    """Return True when the patch parses and applies cleanly to the skill."""
    try:
        apply_patch(skill_md, patch_text)
    except PatchError:
        return False
    return True


def _apply_hunk(body: list[str], hunk: PatchHunk) -> None:
    """Replace the hunk's matched context/removed block in ``body`` in place."""
    needle = hunk.context + hunk.removed
    if not needle:
        raise PatchError(f"patch does not apply: empty hunk for section {hunk.section!r}")
    span = len(needle)
    positions = [
        index
        for index in range(len(body) - span + 1)
        if body[index : index + span] == needle
    ]
    if not positions:
        raise PatchError(f"patch does not apply: not found in section {hunk.section!r}")
    if len(positions) > 1:
        raise PatchError(f"patch does not apply: ambiguous match in section {hunk.section!r}")
    start = positions[0]
    body[start : start + span] = hunk.context + hunk.added


def _split_sections(skill_md: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Split skill markdown into (preamble, [(header_line, body_lines), ...]).

    Lines starting with ``## `` delimit sections; everything before the first
    header is the preamble. Headers and bodies keep their exact text.
    """
    lines = skill_md.splitlines()
    header_indices = [index for index, line in enumerate(lines) if line.startswith("## ")]
    if not header_indices:
        return list(lines), []
    preamble = lines[: header_indices[0]]
    sections: list[tuple[str, list[str]]] = []
    for position, start in enumerate(header_indices):
        end = header_indices[position + 1] if position + 1 < len(header_indices) else len(lines)
        sections.append((lines[start], list(lines[start + 1 : end])))
    return preamble, sections


def _section_names(skill_md: str) -> set[str]:
    """Normalized names of every ``## <Section>`` header in the skill."""
    return {
        _normalize_section(line[3:])
        for line in skill_md.splitlines()
        if line.startswith("## ")
    }


def _normalize_section(name: str) -> str:
    """Section-name comparison key: lowercase, edges trimmed."""
    return name.strip().casefold()
