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
import time
from typing import Any, Optional

from .llm import Proposal
from .react import AssistantMessage, Backend, ReActDriver, ToolCall, Message
from .tools import ToolContext, as_gigachat_tools, build_tool_registry


# GigaChat occasionally drops the first request after a long CV pause
# (server-side idle TCP/TLS disconnect) and sometimes blows past the default
# httpx 30s read timeout on Max-tier completions with tools. Both are
# recoverable; we retry transient httpx errors with exponential backoff.
_TRANSIENT_ERR_NAMES = {
    "ReadError", "ReadTimeout", "ConnectError", "ConnectTimeout",
    "RemoteProtocolError", "ProxyError", "PoolTimeout", "WriteError",
    "WriteTimeout", "TimeoutError",
}


def _is_transient(exc: BaseException) -> bool:
    if type(exc).__name__ in _TRANSIENT_ERR_NAMES:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_transient(cause)
    return False


class _GigaChatBackend:
    """Thin adapter wrapping a `bind_tools`-enabled langchain GigaChat client.

    Retries on transient httpx network errors with exponential backoff. The
    retry lives in the backend (not in ReActDriver) because only providers
    know what "transient" means for their wire protocol.

    Optionally emits a TOKEN_USAGE event per call (when langchain surfaces
    usage_metadata) so the journal can be summed later for cost accounting.
    """

    def __init__(self, llm, max_retries: int = 4,
                 base_backoff_sec: float = 2.0,
                 on_usage: Optional[Any] = None) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._base_backoff = base_backoff_sec
        self._on_usage = on_usage    # callable(dict) or None

    def invoke(self, messages: list[Message]) -> AssistantMessage:
        lc_msgs = [_to_langchain(m) for m in messages]
        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_retries + 1):
            try:
                ai = self._llm.invoke(lc_msgs)
                self._record_usage(ai)
                return _from_langchain(ai)
            except Exception as e:  # noqa: BLE001
                if not _is_transient(e) or attempt >= self._max_retries:
                    raise
                last_exc = e
                time.sleep(self._base_backoff * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("backend invoke loop terminated unexpectedly")

    def _record_usage(self, ai) -> None:
        if self._on_usage is None:
            return
        usage = getattr(ai, "usage_metadata", None)
        if not usage:
            # Fall back to provider-native shape in response_metadata
            rm = getattr(ai, "response_metadata", {}) or {}
            token_usage = rm.get("token_usage") or rm.get("usage") or {}
            if token_usage:
                usage = {
                    "input_tokens": token_usage.get("prompt_tokens",
                                                     token_usage.get("input_tokens", 0)),
                    "output_tokens": token_usage.get("completion_tokens",
                                                     token_usage.get("output_tokens", 0)),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
        if not usage:
            return
        try:
            self._on_usage({
                "input": int(usage.get("input_tokens", 0)),
                "output": int(usage.get("output_tokens", 0)),
                "total": int(usage.get("total_tokens", 0)),
            })
        except Exception:
            pass


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
        timeout: float = 180.0,
        max_retries: int = 4,
    ) -> None:
        from langchain_gigachat import GigaChat
        creds = os.environ.get("GIGACHAT_CREDENTIALS")
        if not creds:
            raise RuntimeError(
                "GIGACHAT_CREDENTIALS env var not set. Export your base64 "
                "client_id:secret before launching.")
        scope = scope or os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
        # `timeout` is forwarded to the underlying httpx client. Default 180s
        # to absorb slow tool-rich completions on Max-tier.
        kwargs = dict(
            credentials=creds, scope=scope, model=model,
            verify_ssl_certs=verify_ssl_certs,
            temperature=temperature,
            profanity_check=False,
            timeout=timeout,
        )
        try:
            self._llm = GigaChat(**kwargs)
        except TypeError:
            # Older langchain-gigachat versions may not accept `timeout=`.
            kwargs.pop("timeout", None)
            self._llm = GigaChat(**kwargs)
        self._max_tool_calls = max_tool_calls
        self._max_wallclock_sec = max_wallclock_sec
        self._max_retries = max_retries

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
        # Route token-usage events through the loop's journal.
        on_usage = lambda u: context.on_event(
            {"event": "TOKEN_USAGE", "iter": context.iteration,
             "input": u["input"], "output": u["output"],
             "total": u["total"]})
        driver = ReActDriver(
            backend=_GigaChatBackend(bound, max_retries=self._max_retries,
                                     on_usage=on_usage),
            tools=tools, context=context,
            max_tool_calls=self._max_tool_calls,
            max_wallclock_sec=self._max_wallclock_sec,
        )
        return driver.run(system_prompt, iteration_prompt, iteration)
