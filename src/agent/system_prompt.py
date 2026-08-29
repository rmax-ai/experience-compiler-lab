"""Fixed system instructions for the execution agent (PYTHON_ARCHITECTURE.md).

The execution agent receives ONLY: system instructions, active skill, task,
tool interfaces. No knowledge content — that is the H3 separation this
constant exists to enforce.
"""

SYSTEM_INSTRUCTIONS = (
    "You are an agent operating enterprise tools. Follow the active skill. "
    "Do not invent tool results. If a required resource is unavailable, "
    "record it and escalate rather than fabricate. Complete the task only "
    "when all requirements are satisfied."
)
