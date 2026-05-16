"""
Yandex AI Studio ReAct backend.

Uses `yandex-cloud-ml-sdk` (the official SDK that wraps the AI Studio /
Foundation Models APIs) to drive YandexGPT under our shared ReAct loop.

Yandex's tool-calling API is roughly OpenAI-shaped: tool definitions are
passed as a list of `{type: "function", function: {name, parameters}}` and
the model returns `tool_calls` inside its response. The SDK details have
churned between versions, so this adapter is defensive: it gracefully
extracts content + tool calls from a few common response shapes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from .llm import Proposal
from .react import AssistantMessage, Backend, Message, ReActDriver, ToolCall
from .tools import ToolContext, as_yandex_tools, build_tool_registry


class _YandexBackend:
    def __init__(self, model_obj, tools_wire: list[dict]) -> None:
        self._model = model_obj
        self._tools_wire = tools_wire

    def invoke(self, messages: list[Message]) -> AssistantMessage:
        sdk_messages = [_to_sdk(m) for m in messages]
        # Newer SDKs accept `tools=` directly; older ones via `.with_tools(...)`.
        try:
            result = self._model.run(sdk_messages, tools=self._tools_wire)
        except TypeError:
            model = self._model
            if hasattr(model, "with_tools"):
                model = model.with_tools(self._tools_wire)
            elif hasattr(model, "configure"):
                model = model.configure(tools=self._tools_wire)
            result = model.run(sdk_messages)
        return _from_sdk(result)


def _to_sdk(m: Message) -> dict:
    # Yandex uses `text` rather than `content`. For tool messages we still
    # send the result text under `text` with a `tool_call_id` annotation.
    if m.role == "system":
        return {"role": "system", "text": m.content}
    if m.role == "user":
        return {"role": "user", "text": m.content}
    if m.role == "tool":
        return {"role": "tool", "text": m.content,
                "tool_call_id": m.tool_call_id or "",
                "name": m.name or ""}
    # assistant
    out: dict = {"role": "assistant", "text": m.content or ""}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name,
                          "arguments": json.dumps(tc.args)}}
            for tc in m.tool_calls
        ]
    return out


def _from_sdk(result: Any) -> AssistantMessage:
    """Pull (text, tool_calls) out of a Yandex SDK completion result.

    The SDK returns an iterable of "alternatives"; we take the first one.
    The exact attribute names vary across versions, so this digs through
    several plausible shapes.
    """
    alts = list(result) if hasattr(result, "__iter__") else [result]
    if not alts:
        return AssistantMessage(content="", raw=result)
    a = alts[0]
    text = (getattr(a, "text", None)
            or getattr(a, "content", None)
            or getattr(a, "message", None)
            or "")
    if not isinstance(text, str):
        text = str(text)

    raw_calls = (getattr(a, "tool_calls", None)
                 or getattr(a, "function_calls", None)
                 or [])
    calls: list[ToolCall] = []
    for tc in raw_calls or []:
        if isinstance(tc, dict):
            tid = tc.get("id") or ""
            fn = tc.get("function", tc)
            name = fn.get("name", "") if isinstance(fn, dict) else ""
            args = fn.get("arguments", fn.get("args", {})) if isinstance(fn, dict) else {}
        else:
            tid = getattr(tc, "id", "") or ""
            fn = getattr(tc, "function", tc)
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", None) or getattr(fn, "args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if name:
            calls.append(ToolCall(id=tid, name=name, args=args or {}))

    return AssistantMessage(content=text or "", tool_calls=calls, raw=result)


class YandexAgent:
    """YandexGPT ReAct backend wired through our tool registry."""

    def __init__(
        self,
        model: str = "yandexgpt",
        model_version: str = "latest",
        max_tool_calls: int = 15,
        max_wallclock_sec: float = 900.0,
        temperature: float = 0.2,
        folder_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from yandex_cloud_ml_sdk import YCloudML
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "yandex-cloud-ml-sdk not installed. "
                "Install via: pip install yandex-cloud-ml-sdk") from e
        folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        api_key = api_key or os.environ.get("YANDEX_API_KEY") \
            or os.environ.get("YC_API_KEY")
        if not folder_id or not api_key:
            raise RuntimeError(
                "Set YANDEX_FOLDER_ID and YANDEX_API_KEY env vars (or pass "
                "folder_id=/api_key= explicitly).")
        self._sdk = YCloudML(folder_id=folder_id, auth=api_key)
        completions = self._sdk.models.completions(model,
                                                    model_version=model_version)
        self._model = completions.configure(temperature=temperature)
        self._max_tool_calls = max_tool_calls
        self._max_wallclock_sec = max_wallclock_sec

    def propose(self, system_prompt: str, iteration_prompt: str,
                iteration: int, context: Optional[Any] = None) -> Proposal:
        if context is None:
            raise RuntimeError("YandexAgent requires a ToolContext via "
                               "`context=`. Run it through ResearchLoop.")
        tools = build_tool_registry()
        wire = as_yandex_tools(tools)
        driver = ReActDriver(
            backend=_YandexBackend(self._model, wire),
            tools=tools, context=context,
            max_tool_calls=self._max_tool_calls,
            max_wallclock_sec=self._max_wallclock_sec,
        )
        return driver.run(system_prompt, iteration_prompt, iteration)
