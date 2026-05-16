"""
GigaChat ReAct backend.

Talks to GigaChat via `langchain-gigachat` and drives a multi-turn tool-use
loop through the shared `ReActDriver`. The model decides when it has seen
enough of the data and calls `submit_solution` to commit a new solution.py;
the driver then closes the iteration and hands the Proposal back to
ResearchLoop for CV-evaluation.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .llm import Proposal
from .react import AssistantMessage, Backend, ReActDriver, ToolCall, Message
from .tools import ToolContext, as_gigachat_tools, build_tool_registry


class _GigaChatBackend:
    """Thin adapter wrapping a `bind_tools`-enabled langchain GigaChat client."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def invoke(self, messages: list[Message]) -> AssistantMessage:
        lc_msgs = [_to_langchain(m) for m in messages]
        ai = self._llm.invoke(lc_msgs)
        return _from_langchain(ai)


def _to_langchain(m: Message):
    from langchain_core.messages import (AIMessage, HumanMessage,
                                          SystemMessage, ToolMessage)
    if m.role == "system":
        return SystemMessage(content=m.content)
    if m.role == "user":
        return HumanMessage(content=m.content)
    if m.role == "tool":
        return ToolMessage(content=m.content, tool_call_id=m.tool_call_id or "",
                           name=m.name or "")
    # assistant
    ai = AIMessage(content=m.content or "")
    if m.tool_calls:
        # LangChain accepts the OpenAI-style schema on construction.
        ai.tool_calls = [
            {"id": tc.id, "name": tc.name, "args": tc.args, "type": "tool_call"}
            for tc in m.tool_calls
        ]
    return ai


def _from_langchain(ai) -> AssistantMessage:
    content = getattr(ai, "content", "") or ""
    raw_calls = getattr(ai, "tool_calls", []) or []
    calls: list[ToolCall] = []
    for tc in raw_calls:
        # LangChain returns either dict-style or pydantic-typed calls.
        if isinstance(tc, dict):
            tid = tc.get("id") or tc.get("tool_call_id") or ""
            name = tc.get("name") or tc.get("function", {}).get("name") or ""
            args = tc.get("args") or tc.get("arguments") \
                   or tc.get("function", {}).get("arguments") or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
        else:
            tid = getattr(tc, "id", "") or ""
            name = getattr(tc, "name", "") or ""
            args = getattr(tc, "args", {}) or {}
        calls.append(ToolCall(id=tid, name=name, args=args))
    if isinstance(content, list):
        # langchain sometimes returns content as a list of blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                          for b in content)
    return AssistantMessage(content=content, tool_calls=calls, raw=ai)


class GigaChatAgent:
    """LangChain-GigaChat backend driving a ReAct loop with our tool registry."""

    def __init__(
        self,
        model: str = "GigaChat-2-Max",
        scope: Optional[str] = None,
        max_tool_calls: int = 15,
        max_wallclock_sec: float = 900.0,
        temperature: float = 0.2,
        verify_ssl_certs: bool = False,
    ) -> None:
        from langchain_gigachat import GigaChat
        creds = os.environ.get("GIGACHAT_CREDENTIALS")
        if not creds:
            raise RuntimeError(
                "GIGACHAT_CREDENTIALS env var not set. Export your base64 "
                "client_id:secret before launching.")
        scope = scope or os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
        self._llm = GigaChat(
            credentials=creds, scope=scope, model=model,
            verify_ssl_certs=verify_ssl_certs,
            temperature=temperature,
            profanity_check=False,
        )
        self._max_tool_calls = max_tool_calls
        self._max_wallclock_sec = max_wallclock_sec

    def propose(self, system_prompt: str, iteration_prompt: str,
                iteration: int, context: Optional[Any] = None) -> Proposal:
        if context is None:
            raise RuntimeError("GigaChatAgent requires a ToolContext via "
                               "`context=`. Run it through ResearchLoop.")
        tools = build_tool_registry()
        wire = as_gigachat_tools(tools)
        try:
            bound = self._llm.bind_tools(wire)
        except TypeError:
            # Older versions used a different signature; fall back.
            bound = self._llm.bind(tools=wire)
        driver = ReActDriver(
            backend=_GigaChatBackend(bound),
            tools=tools, context=context,
            max_tool_calls=self._max_tool_calls,
            max_wallclock_sec=self._max_wallclock_sec,
        )
        return driver.run(system_prompt, iteration_prompt, iteration)
