"""LLM agent adapter: drives a tool-calling model through the benchmark.

The benchmark's step interface yields one action at a time, while a model turn
can contain several tool calls. ``LLMAgent`` bridges the two: it queues the
``tool_use`` blocks of a turn, executes them one by one, then returns *all*
``tool_result`` blocks in a single user message (splitting them discourages
parallel tool use). Assistant turns are appended with their full ``content``
(thinking blocks included, unchanged) as the API requires.

Backends implement ``ChatBackend.create(system, messages, tools)`` and return
an object with ``.content`` (blocks with ``.type``/``.text``/``.id``/``.name``/
``.input``) and ``.stop_reason`` — the Anthropic Messages response shape.
``AnthropicBackend`` is provided; tests use a scripted fake, so CI needs no key.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .runner import Action

SYSTEM_PROMPT = """You are a customer-operations agent for an online retailer.
Policy:
- Authenticate the user (by email, or by full name + zip code) before accessing
  any account or order data.
- Only access orders that belong to the authenticated user. Never reveal another customer's data.
- Only pending orders can be cancelled (reason: 'no longer needed' or 'ordered by mistake').
- Only delivered orders within 30 days of delivery can be returned; refunds go to
  'original' or 'gift_card'.
- Before any change (cancel, return, address update), describe it and get explicit
  confirmation from the user.
- If a request violates policy, explain that you cannot do it.
- If a tool returns a transient 503 error, retry the call.
- Tell the user the outcome, including any refund amount."""


class ChatBackend(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> Any: ...


def to_anthropic_tools(schemas: list[dict]) -> list[dict]:
    return [
        {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
        for s in schemas
    ]


class AnthropicBackend:
    """Claude via the official SDK, with server-side refusal fallback enabled."""

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000):
        import anthropic  # optional dependency: pip install -e ".[anthropic]"

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def create(self, system, messages, tools):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )


class LLMAgent:
    def __init__(self, backend: ChatBackend, name: str = "llm", system: str = SYSTEM_PROMPT):
        self.backend = backend
        self.name = name
        self.system = system

    def reset(self, tools: list[dict], seed: int) -> None:
        self.tools = to_anthropic_tools(tools)
        self.messages: list[dict] = []
        self.seen = 0
        self.pending: list[Any] = []  # tool_use blocks not yet executed
        self.in_flight: Any = None
        self.results: list[dict] = []
        self.usage_calls = 0

    def _sync(self, history: list[dict]) -> None:
        for h in history[self.seen :]:
            if h["role"] == "user":
                self.messages.append({"role": "user", "content": h["content"]})
            elif h["role"] == "tool" and self.in_flight is not None:
                out = h["content"]
                self.results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": self.in_flight.id,
                        "content": out,
                        "is_error": '"error"' in out,
                    }
                )
                self.in_flight = None
        self.seen = len(history)

    def act(self, history: list[dict]) -> Action:
        self._sync(history)
        if self.pending:
            return self._dispatch()
        if self.results:
            self.messages.append({"role": "user", "content": self.results})
            self.results = []
        resp = self.backend.create(self.system, self.messages, self.tools)
        self.usage_calls += 1
        self.messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            return Action("end")
        self.pending = [b for b in resp.content if b.type == "tool_use"]
        if self.pending:
            return self._dispatch()
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        return Action("message", text=text) if text else Action("end")

    def _dispatch(self) -> Action:
        block = self.pending.pop(0)
        self.in_flight = block
        args = block.input if isinstance(block.input, dict) else json.loads(block.input)
        return Action("tool", block.name, args)
