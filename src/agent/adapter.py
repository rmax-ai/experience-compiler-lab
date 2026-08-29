"""Minimal OpenAI-compatible LLM adapter plus a scripted test double.

``LlmAdapter`` talks to any OpenAI-compatible ``/chat/completions`` endpoint
over httpx (120s timeout, 2 retries with 2s/4s backoff on 429/5xx).
``FakeModel`` is the default when ``EXP_LLM_API_KEY`` is unset: it serves
scripted responses (from a JSONL file or an inline list) so the harness runs
deterministically without network access.

H3 rule: this module (and everything else under ``agent/``) never imports
``knowledge`` — execution-time context must not see learning-time state.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

_PRICING_PATH = Path(__file__).resolve().parent / "pricing.yaml"
_PRICES: dict[str, dict[str, float]] | None = None


def _load_prices() -> dict[str, dict[str, float]]:
    """Load the example pricing table once (per-1M-token USD)."""
    global _PRICES
    if _PRICES is None:
        data = yaml.safe_load(_PRICING_PATH.read_text(encoding="utf-8"))
        _PRICES = {str(name): dict(entry) for name, entry in (data or {}).items()}
    return _PRICES


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Cost of one call: usage × per-1M-token prices, defaulted by model name."""
    entry = _load_prices().get(model) or _load_prices().get("default", {})
    input_price = float(entry.get("input", 0.30))
    output_price = float(entry.get("output", 1.20))
    return input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price


@dataclass(frozen=True)
class Usage:
    """Token usage and estimated cost of one completion call."""

    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class CompletionResult:
    """One chat-completions response: the assistant message plus usage."""

    message: dict[str, Any]
    usage: Usage


class LlmAdapter:
    """OpenAI-compatible chat completions client with bounded retries."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> LlmAdapter:
        """Build from ``EXP_LLM_BASE_URL`` / ``EXP_LLM_API_KEY`` / ``EXP_LLM_MODEL``."""
        base_url = os.environ.get("EXP_LLM_BASE_URL", "")
        api_key = os.environ.get("EXP_LLM_API_KEY", "")
        model = os.environ.get("EXP_LLM_MODEL", "")
        missing = [
            name
            for name, value in (
                ("EXP_LLM_BASE_URL", base_url),
                ("EXP_LLM_API_KEY", api_key),
                ("EXP_LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "LlmAdapter requires env vars for the LLM endpoint; missing: "
                + ", ".join(missing)
            )
        return cls(base_url=base_url, api_key=api_key, model=model)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """POST /chat/completions with 2 retries (2s/4s) on 429/5xx.

        Raises a clear ``RuntimeError`` (with the HTTP status) once retries
        are exhausted.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        backoffs = (2.0, 4.0)
        last_status: int | None = None

        with httpx.Client(timeout=120.0) as client:
            for attempt in range(len(backoffs) + 1):
                try:
                    response = client.post(url, json=payload, headers=headers)
                except httpx.RequestError as exc:
                    if attempt >= len(backoffs):
                        raise RuntimeError(
                            f"LLM request failed after retries (network error): {exc}"
                        ) from exc
                    time.sleep(backoffs[attempt])
                    continue

                status = response.status_code
                if status == 429 or status >= 500:
                    last_status = status
                    if attempt >= len(backoffs):
                        raise RuntimeError(
                            f"LLM request failed after retries: HTTP {last_status}"
                        )
                    time.sleep(backoffs[attempt])
                    continue
                if status >= 400:
                    raise RuntimeError(
                        f"LLM request failed: HTTP {status}: {response.text[:200]}"
                    )
                return self._parse_response(response.json())

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> CompletionResult:
        choices = data.get("choices") or []
        if not choices or "message" not in choices[0]:
            raise RuntimeError(f"malformed LLM response: {data}")
        usage_raw = data.get("usage") or {}
        input_tokens = int(usage_raw.get("prompt_tokens", 0))
        output_tokens = int(usage_raw.get("completion_tokens", 0))
        model = str(data.get("model", ""))
        return CompletionResult(
            message=choices[0]["message"],
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens, model),
            ),
        )


class FakeModel:
    """Scripted test double with the same ``complete`` interface as LlmAdapter.

    Scripts come from either a JSONL file (each line:
    ``{"messages": [...], "response": {...}, "usage": {...}}``) or an inline
    list of ``(response, usage)`` tuples. File scripts are matched by the
    *last user message* text (the task description); the first matching script
    is popped. Inline scripts are popped in order and always match.
    """

    def __init__(
        self,
        scripts: str | Path | list[tuple[dict[str, Any], dict[str, int]]] | None = None,
        *,
        model: str = "fake",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._scripts: list[dict[str, Any]] = []
        if scripts is None:
            return
        if isinstance(scripts, (str, Path)):
            self._load_scripts(Path(scripts))
        else:
            for response, usage in scripts:
                self._scripts.append(
                    {"inline": True, "response": response, "usage": usage}
                )

    def _load_scripts(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid script at line {lineno} of {path}: {exc}"
                    ) from exc
                entry.setdefault("messages", [])
                entry.setdefault("usage", {})
                self._scripts.append(entry)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        last_user = self._last_user_text(messages)
        for index, script in enumerate(self._scripts):
            if script.get("inline") or self._last_user_text(script.get("messages", [])) == last_user:
                entry = self._scripts.pop(index)
                return self._to_result(entry)
        if not self._scripts:
            raise RuntimeError("FakeModel script exhausted")
        raise RuntimeError(f"no scripted response for last user message: {last_user!r}")

    def _to_result(self, entry: dict[str, Any]) -> CompletionResult:
        usage_raw = entry.get("usage") or {}
        input_tokens = int(usage_raw.get("input_tokens", 0))
        output_tokens = int(usage_raw.get("output_tokens", 0))
        return CompletionResult(
            message=entry["response"],
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens, self.model),
            ),
        )

    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return ""
