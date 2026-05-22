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
from .tools import ToolContext, build_tool_registry


class _YandexBackend:
    def __init__(self, model_obj, sdk_tools: list,
                 on_usage: Optional[Any] = None) -> None:
        # `sdk_tools` is a list of `FunctionTool` proto-wrapped objects
        # built via `sdk.tools.function(...)`. Plain dicts cause
        # `'dict' object has no attribute '_to_proto'` inside the SDK.
        self._model = model_obj
        self._sdk_tools = sdk_tools
        self._on_usage = on_usage

    def invoke(self, messages: list[Message]) -> AssistantMessage:
        sdk_messages = [_to_sdk(m) for m in messages]
        try:
            result = self._model.run(sdk_messages, tools=self._sdk_tools)
        except TypeError:
            model = self._model
            if hasattr(model, "with_tools"):
                model = model.with_tools(self._sdk_tools)
            elif hasattr(model, "configure"):
                model = model.configure(tools=self._sdk_tools)
            result = model.run(sdk_messages)
        self._record_usage(result)
        return _from_sdk(result)

    def _record_usage(self, result) -> None:
        if self._on_usage is None:
            return
        usage = getattr(result, "usage", None)
        if usage is None:
            return
        # Yandex CompletionUsage: input_text_tokens, completion_tokens,
        # total_tokens, reasoning_tokens.
        try:
            self._on_usage({
                "input": int(getattr(usage, "input_text_tokens", 0) or 0),
                "output": int(getattr(usage, "completion_tokens", 0) or 0),
                "total": int(getattr(usage, "total_tokens", 0) or 0),
            })
        except Exception:
            pass


class _YandexAssistantToolCallMessage:
    """Satisfies the SDK's TextMessageWithToolCallsProtocol.

    Yandex requires structured assistant-with-tool-calls echoed back when
    the next message carries tool_results. We pass the SDK-native
    `ToolCallList` we got from the model's response — its `_proto_origin`
    field is what the SDK reads when serializing.
    """

    def __init__(self, text: str, raw_tool_calls: Any):
        self.role = "assistant"
        self.text = text
        self.tool_calls = raw_tool_calls   # ToolCallList from a prior response


def _to_sdk(m: Message):
    """Translate our normalized Message into Yandex SDK's input type.

    Yandex's protocol is NOT OpenAI-shaped:
      - No `role="tool"` — tool results travel in a separate message dict
        `{"tool_results": [{"name", "content", "type": "function"}, ...]}`.
      - Assistant messages with tool calls require a typed object
        (TextMessageWithToolCallsProtocol). We pass the original SDK
        ToolCallList from `raw_tool_calls` — plain text would make Yandex
        reject the tool_results that follow ("no prior tool call").
      - Empty `text` is rejected with `INVALID_ARGUMENT: empty message text`,
        so we always send at least a space.
    """
    text = (m.content or "").strip()
    if m.role == "tool":
        return {
            "role": "assistant",
            "tool_results": [{
                "name": m.name or "tool",
                "content": text or " ",
                "type": "function",
            }],
        }
    if m.role == "assistant" and m.tool_calls:
        if m.raw_tool_calls is not None:
            return _YandexAssistantToolCallMessage(
                text=text or " ", raw_tool_calls=m.raw_tool_calls)
        rendered = text + ("\n\n" if text else "") + "\n".join(
            f"[tool_call] {tc.name}({json.dumps(tc.args, ensure_ascii=False)})"
            for tc in m.tool_calls)
        return {"role": "assistant", "text": rendered or " "}
    if m.role == "system":
        return {"role": "system", "text": text or " "}
    if m.role == "user":
        return {"role": "user", "text": text or " "}
    return {"role": "assistant", "text": text or " "}


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

    # Preserve SDK-native ToolCallList so we can echo the assistant turn back
    # in the proper Yandex format (see _YandexAssistantToolCallMessage).
    raw_tcl = raw_calls if calls else None
    return AssistantMessage(content=text or "", tool_calls=calls, raw=result,
                            raw_tool_calls=raw_tcl)


class YandexAgent:
    """YandexGPT ReAct backend wired through our tool registry."""

    def __init__(
        self,
        model: str = "yandexgpt",
        model_version: str = "latest",
        max_tool_calls: int = 64,
        max_wallclock_sec: float = 900.0,
        temperature: float = 0.6,
        folder_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        # The SDK talks gRPC over TLS. On Windows with the system Python,
        # gRPC ships without a default CA bundle and fails with
        # CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundle before the
        # SDK initializes its channel — this is a no-op on systems where
        # gRPC already finds a CA path.
        if "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH" not in os.environ:
            try:
                import certifi
                os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()
            except Exception:
                pass

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
        self._model_name = model    # raw short name for pricing lookup
        self._max_tool_calls = max_tool_calls
        self._max_wallclock_sec = max_wallclock_sec

    def propose(self, system_prompt: str, iteration_prompt: str,
                iteration: int, context: Optional[Any] = None) -> Proposal:
        if context is None:
            raise RuntimeError("YandexAgent requires a ToolContext via "
                               "`context=`. Run it through ResearchLoop.")
        tools = build_tool_registry()
        # Wrap each ToolSpec in the SDK's FunctionTool so model.run accepts
        # tools=[...]; raw dicts fail with `'dict' has no attribute _to_proto`.
        sdk_tools = [self._sdk.tools.function(
            parameters=t.json_schema, name=t.name, description=t.description)
            for t in tools]
        from .pricing import cost_rub as _cost_rub
        def on_usage(u):
            ev = {"event": "TOKEN_USAGE", "iter": context.iteration,
                  "model": self._model_name,
                  "input": u["input"], "output": u["output"],
                  "total": u["total"]}
            c = _cost_rub(self._model_name, u["input"], u["output"])
            if c is not None:
                ev["cost_rub"] = round(c, 6)
            context.on_event(ev)
        driver = ReActDriver(
            backend=_YandexBackend(self._model, sdk_tools, on_usage=on_usage),
            tools=tools, context=context,
            max_tool_calls=self._max_tool_calls,
            max_wallclock_sec=self._max_wallclock_sec,
        )
        return driver.run(system_prompt, iteration_prompt, iteration)
