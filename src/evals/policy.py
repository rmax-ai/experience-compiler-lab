"""Promotion policy (SPEC.md §10): a pure function over the eval result.

The propose step never decides; ``promote`` applies this policy. Pure: no IO,
no clocks, no hidden state — identical inputs always yield the same decision.
"""

from __future__ import annotations

from evals.runner import SkillEvalResult


def decide(eval_result: SkillEvalResult, allowed_regressions: int = 0) -> bool:
    """Simplest promotion rule (SPEC.md §10).

    Accept iff the candidate beats the baseline success rate on the fixed
    validation split AND regressions (tasks the baseline passed but the
    candidate failed) stay within ``allowed_regressions``. The score vector
    is recorded for evidence but deliberately NOT collapsed into the decision
    yet (SPEC.md §10: "Do not collapse everything into a single score").
    """
    return (
        eval_result.candidate_success_rate > eval_result.baseline_success_rate
        and len(eval_result.regressions) <= allowed_regressions
    )


def decision_reason(eval_result: SkillEvalResult, allowed_regressions: int = 0) -> str:
    """One-line explanation of the decision for the append-only ledger."""
    return (
        f"{eval_result.baseline_success_rate:.2f} -> "
        f"{eval_result.candidate_success_rate:.2f}, regressions "
        f"{len(eval_result.regressions)}/{allowed_regressions}"
    )
