"""Promotion policy tests (SPEC.md §10): pure decision function semantics."""

from evals.policy import decide, decision_reason
from evals.runner import SkillEvalResult


def _result(
    baseline: float, candidate: float, regressions: list[str] | None = None
) -> SkillEvalResult:
    return SkillEvalResult(
        baseline_success_rate=baseline,
        candidate_success_rate=candidate,
        regressions=regressions or [],
        score_vector_delta={},
        baseline_by_task={},
        candidate_by_task={},
    )


def test_improvement_without_regressions_accepts() -> None:
    assert decide(_result(0.7, 0.8), allowed_regressions=0) is True


def test_equal_rates_reject() -> None:
    assert decide(_result(0.7, 0.7), allowed_regressions=0) is False


def test_regression_above_allowance_rejects() -> None:
    result = _result(0.7, 0.8, regressions=["task-a"])
    assert decide(result, allowed_regressions=0) is False


def test_regression_within_allowance_accepts() -> None:
    result = _result(0.7, 0.8, regressions=["task-a"])
    assert decide(result, allowed_regressions=1) is True


def test_decision_reason_one_line() -> None:
    result = _result(0.7, 0.8, regressions=["task-a"])
    assert decision_reason(result, allowed_regressions=1) == "0.70 -> 0.80, regressions 1/1"
