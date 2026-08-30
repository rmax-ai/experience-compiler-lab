"""Configuration definitions for the M5 ablation experiments."""

from __future__ import annotations

from enum import StrEnum


class AblationConfig(StrEnum):
    """Named configurations in the M5 experimental baseline matrix."""

    BASELINE = "baseline"
    TRACE2SKILL = "trace2skill"
    MEMORY = "memory"
    COMPILER = "compiler"


CONFIG_MATRIX: dict[str, dict[str, bool]] = {
    AblationConfig.BASELINE.value: {
        "persist_traces": False,
        "mine_knowledge": False,
        "evolve_skills": False,
        "knowledge_in_execution": False,
    },
    AblationConfig.TRACE2SKILL.value: {
        "persist_traces": True,
        "mine_knowledge": False,
        "evolve_skills": True,
        "knowledge_in_execution": False,
    },
    AblationConfig.MEMORY.value: {
        "persist_traces": True,
        "mine_knowledge": True,
        "evolve_skills": False,
        "knowledge_in_execution": True,
    },
    AblationConfig.COMPILER.value: {
        "persist_traces": True,
        "mine_knowledge": True,
        "evolve_skills": True,
        "knowledge_in_execution": False,
    },
}


def config_help() -> str:
    """Return concise descriptions of every available ablation configuration."""
    return "\n".join(
        (
            "baseline: no persistent traces, structured knowledge, or evolved skills.",
            "trace2skill: persistent traces and evolved skills, without structured knowledge.",
            "memory: persistent traces and structured knowledge supplied to execution as notes.",
            "compiler: persistent traces, structured knowledge, and evolved skills.",
        )
    )
