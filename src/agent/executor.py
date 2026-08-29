"""Tiny deterministic agent loop (SPEC.md §18 "minimal agent loop").

The executor receives ONLY the assembled context (system + skill + task +
tool schemas) plus a :class:`world.state.World` and a model. It never sees
``knowledge/`` (H3). Tool execution never raises: every tool result,
including failures, is captured in the trace as ``{"ok": false, ...}``.

Outcome semantics: the executor fills ``Outcome`` only with transport-level
errors (model call failures, step-budget exhaustion). ``Outcome.success`` and
grader errors are set by ``experiments.runner`` AFTER grading — the grader
decides success, not the executor.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.adapter import FakeModel, LlmAdapter
from agent.context import ExecutionContext
from traces.schema import Action, Manifest, Message, Metrics, Outcome, Scenario, Trace
from world.api import execute as execute_tool
from world.state import Document, Employee, Policy, World, utcnow


def build_world(task: Scenario) -> World:
    """Construct a fresh :class:`World` from a scenario's ``world`` block.

    The resulting world's ``reset()`` restores exactly this pristine state,
    which is what ``Executor.run`` relies on for per-task isolation.
    """
    employees = {emp_id: Employee(**data) for emp_id, data in task.world.employees.items()}
    policies = {
        role: Policy(access_rules=data.get("access_rules", {}))
        for role, data in task.world.policies.items()
    }
    documents = {doc_id: Document(**data) for doc_id, data in task.world.documents.items()}
    return World(
        inventory=dict(task.world.inventory),
        employees=employees,
        policies=policies,
        documents=documents,
        workflows={name: list(steps) for name, steps in task.world.workflows.items()},
    )


def _normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Reduce a model response message to the Message contract fields.

    OpenAI-compatible APIs may return extra keys and ``content: null`` on
    tool-call messages; the trace transcript stores exactly
    role/content/tool_calls (docs/data-formats.md §2).
    """
    content = message.get("content")
    return {
        "role": str(message.get("role", "assistant")),
        "content": content if isinstance(content, str) else "",
        "tool_calls": message.get("tool_calls") or None,
    }


def _tool_result_message(result: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-style ``role: tool`` message echoing the tool result."""
    return {"role": "tool", "content": json.dumps(result, sort_keys=True)}


class Executor:
    """Run one task against the world until a final answer or budget exhaustion."""

    def __init__(
        self,
        model: LlmAdapter | FakeModel,
        world: World,
        max_steps: int = 20,
    ) -> None:
        self.model = model
        self.world = world
        self.max_steps = max_steps

    def run(
        self,
        task: Scenario,
        context: ExecutionContext,
        seed: int,
        run_id: str,
        manifest: Manifest,
        skill_version: str | None = None,
    ) -> Trace:
        """Execute ``task`` and return a completed :class:`Trace`.

        The world must have been constructed from this scenario (e.g. via
        :func:`build_world`); it is reset to that pristine state first.
        ``seed`` is recorded in ``manifest`` by the caller and is accepted for
        interface stability; the loop itself is fully deterministic.
        ``skill_version`` defaults to ``manifest.skill_version`` when omitted.
        """
        self.world.reset()

        system_content = f"{context.system}\n\n## Active skill\n{context.skill}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task.description},
        ]

        actions: list[Action] = []
        tokens_in = 0
        tokens_out = 0
        cost_usd = 0.0
        latency_s = 0.0
        transport_errors: list[str] = []
        final_answer: str | None = None

        for _ in range(self.max_steps):
            started = time.perf_counter()
            try:
                result = self.model.complete(messages, tools=context.tools)
            except Exception as exc:  # noqa: BLE001 — transport errors are evidence
                transport_errors.append(f"model call failed: {exc}")
                final_answer = None
                break
            latency_s += time.perf_counter() - started
            tokens_in += result.usage.input_tokens
            tokens_out += result.usage.output_tokens
            cost_usd += result.usage.estimated_cost_usd

            message = _normalize_assistant_message(result.message)
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content")
                final_answer = content if content else None
                break

            for tool_call in tool_calls:
                action = self._execute_tool_call(tool_call, len(actions))
                actions.append(action)
                messages.append(_tool_result_message(action.result))
        else:
            # Step budget exhausted without a final answer.
            final_answer = None
            transport_errors.append("step budget exhausted")

        trace = Trace(
            run_id=run_id,
            task_id=task.task_id,
            model=self.model.model,
            skill_version=skill_version or manifest.skill_version,
            messages=[Message.model_validate(m) for m in messages],
            actions=actions,
            final_answer=final_answer,
            outcome=Outcome(
                success=False,
                errors=sorted(transport_errors),
                violated_constraints=[],
            ),
            metrics=Metrics(
                tool_calls=len(actions),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                estimated_cost_usd=round(cost_usd, 8),
                latency_s=round(latency_s, 4),
                recovery_count=self._recovery_count(actions),
                trajectory_length=len(messages),
            ),
            manifest=manifest,
        )
        return trace

    def _execute_tool_call(
        self, tool_call: dict[str, Any], index: int
    ) -> Action:
        """Execute one tool call; never raises (failures become ok=false)."""
        function = tool_call.get("function", {})
        tool = str(function.get("name", ""))
        arguments: dict[str, Any] = {}
        parse_error: dict[str, Any] | None = None
        try:
            parsed = json.loads(function.get("arguments") or "{}")
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"arguments must be a JSON object, got {type(parsed).__name__}"
                )
            arguments = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = {"ok": False, "error": f"invalid arguments: {exc}"}

        before = self.world.snapshot()
        timestamp = utcnow()
        if parse_error is not None:
            tool_result = parse_error
        else:
            try:
                tool_result = execute_tool(self.world, tool, arguments)
            except Exception as exc:  # noqa: BLE001 — tool failures are evidence
                tool_result = {"ok": False, "error": str(exc)}
        after = self.world.snapshot()

        return Action(
            index=index,
            tool=tool,
            arguments=arguments,
            result=tool_result,
            timestamp=timestamp,
            world_state_before=before,
            world_state_after=after,
        )

    @staticmethod
    def _recovery_count(actions: list[Action]) -> int:
        """Failures later retried with the same tool successfully (SPEC §10)."""
        return sum(
            1
            for i, action in enumerate(actions)
            if action.result.get("ok") is False
            and any(
                later.tool == action.tool and later.result.get("ok") is True
                for later in actions[i + 1 :]
            )
        )
