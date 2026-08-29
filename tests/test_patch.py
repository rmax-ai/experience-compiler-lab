"""Patch grammar tests: parsing, deterministic application, ambiguity."""

import pytest

from skills.patch import PatchError, apply_patch, parse_patch, verify_applies

SKILL = """# Employee Onboarding

## Objective
Complete onboarding while respecting workflow dependencies.

## Procedure
1. Resolve employee identity.
2. Retrieve onboarding requirements.
3. Check hardware inventory.
4. Assign available hardware.
5. Request procurement if inventory is unavailable.

## Failure handling
If a required dependency cannot be fulfilled:
- record the blocker;
- escalate to the responsible team;
- leave the workflow incomplete.
"""

_REPLACE_PATCH = (
    "@@ Procedure\n"
    "- 4. Assign available hardware.\n"
    "+ 4. Assign hardware only if inventory confirms availability.\n"
)


# -- parse_patch --------------------------------------------------------------


def test_parse_rejects_unknown_section() -> None:
    with pytest.raises(PatchError, match="unknown section"):
        parse_patch("@@ ProcedureX\n- 4. Assign available hardware.\n+ x.\n", SKILL)


def test_parse_rejects_junk_lines() -> None:
    with pytest.raises(PatchError, match="junk"):
        parse_patch("some prose before any hunk\n@@ Procedure\n- x\n+ y\n", SKILL)
    with pytest.raises(PatchError, match="malformed"):
        parse_patch("@@ Procedure\n-x\n", SKILL)
    with pytest.raises(PatchError, match="malformed"):
        parse_patch("@@ Procedure\n@@@ weird\n", SKILL)


def test_parse_rejects_empty_patch() -> None:
    with pytest.raises(PatchError, match="empty patch"):
        parse_patch("", SKILL)
    with pytest.raises(PatchError, match="empty patch"):
        parse_patch("\n  \n\t\n", SKILL)


def test_parse_rejects_empty_hunk() -> None:
    with pytest.raises(PatchError, match="empty hunk"):
        parse_patch("@@ Procedure\n2. Retrieve onboarding requirements.\n", SKILL)


def test_parse_grammar_without_skill() -> None:
    hunks = parse_patch(_REPLACE_PATCH)
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.section == "Procedure"
    assert hunk.removed == ["4. Assign available hardware."]
    assert hunk.added == ["4. Assign hardware only if inventory confirms availability."]
    assert hunk.context == []


def test_parse_context_anchors() -> None:
    hunks = parse_patch(
        "@@ Procedure\n"
        "2. Retrieve onboarding requirements.\n"
        "- 3. Check hardware inventory.\n"
        "+ 3. Check available inventory.\n",
        SKILL,
    )
    assert hunks[0].context == ["2. Retrieve onboarding requirements."]
    assert hunks[0].removed == ["3. Check hardware inventory."]


# -- apply_patch -------------------------------------------------------------


def test_apply_single_line_replacement() -> None:
    result = apply_patch(SKILL, _REPLACE_PATCH)
    assert "4. Assign hardware only if inventory confirms availability." in result
    assert "4. Assign available hardware." not in result
    # Preamble and headers preserved verbatim.
    assert result.startswith("# Employee Onboarding\n\n## Objective\n")
    assert "## Procedure\n" in result
    assert "## Failure handling\n" in result


def test_apply_multi_line_addition() -> None:
    patch = (
        "@@ Procedure\n"
        "- 4. Assign available hardware.\n"
        "+ 4. Assign hardware only if inventory confirms availability.\n"
        "+ 4b. Otherwise create a procurement blocker.\n"
    )
    result = apply_patch(SKILL, patch)
    assert "4. Assign hardware only if inventory confirms availability." in result
    assert "4b. Otherwise create a procurement blocker." in result


def test_apply_removal_only() -> None:
    result = apply_patch(SKILL, "@@ Procedure\n- 4. Assign available hardware.\n")
    assert "4. Assign available hardware." not in result
    assert "5. Request procurement if inventory is unavailable." in result


def test_apply_multiple_hunks() -> None:
    patch = (
        "@@ Procedure\n"
        "- 3. Check hardware inventory.\n"
        "+ 3. Check available inventory.\n"
        "@@ Failure handling\n"
        "- - leave the workflow incomplete.\n"
        "+ - leave the workflow incomplete until the blocker clears.\n"
    )
    result = apply_patch(SKILL, patch)
    assert "3. Check available inventory." in result
    assert "3. Check hardware inventory." not in result
    assert "- leave the workflow incomplete until the blocker clears." in result
    assert "leave the workflow incomplete." not in result


def test_apply_section_names_case_insensitive() -> None:
    result = apply_patch(SKILL, _REPLACE_PATCH.replace("@@ Procedure", "@@ procedure"))
    assert "4. Assign hardware only if inventory confirms availability." in result


def test_apply_deterministic() -> None:
    assert apply_patch(SKILL, _REPLACE_PATCH) == apply_patch(SKILL, _REPLACE_PATCH)


def test_apply_context_anchored_with_repeated_lines() -> None:
    repeated = (
        "# Demo Skill\n"
        "## Steps\n"
        "Repeat the verification step:\n"
        "- verify all mandatory tasks;\n"
        "- verify all mandatory tasks;\n"
    )
    patch = (
        "@@ Steps\n"
        "Repeat the verification step:\n"
        "- - verify all mandatory tasks;\n"
        "+ - verify every mandatory task;\n"
    )
    result = apply_patch(repeated, patch)
    assert "- verify every mandatory task;" in result
    assert result.count("verify all mandatory tasks;") == 1


def test_apply_ambiguous_match_raises() -> None:
    repeated = (
        "# Demo Skill\n"
        "## Steps\n"
        "Repeat the verification step:\n"
        "- verify all mandatory tasks;\n"
        "- verify all mandatory tasks;\n"
    )
    with pytest.raises(PatchError, match="ambiguous"):
        apply_patch(repeated, "@@ Steps\n- - verify all mandatory tasks;\n+ x;\n")


def test_apply_not_found_raises() -> None:
    with pytest.raises(PatchError, match="not found"):
        apply_patch(SKILL, "@@ Procedure\n- this line does not exist.\n+ x.\n")


def test_failed_apply_leaves_no_partial_state() -> None:
    # The first hunk would apply on its own; the second cannot. apply_patch
    # must raise instead of returning a partially-applied skill.
    patch = (
        "@@ Procedure\n"
        "- 4. Assign available hardware.\n"
        "+ 4. Assign hardware only if inventory confirms availability.\n"
        "@@ Failure handling\n"
        "- this line does not exist in the skill.\n"
        "+ replacement.\n"
    )
    with pytest.raises(PatchError, match="not found"):
        apply_patch(SKILL, patch)
    assert not verify_applies(SKILL, patch)


# -- verify_applies -----------------------------------------------------------


def test_verify_applies() -> None:
    assert verify_applies(SKILL, _REPLACE_PATCH) is True
    assert verify_applies(SKILL, "@@ Procedure\n- missing line.\n+ x.\n") is False
    assert verify_applies(SKILL, "@@ NoSuchSection\n- x.\n+ y.\n") is False
