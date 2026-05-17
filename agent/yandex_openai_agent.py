"""
Yandex AI Studio OpenAI-compatible backend.

A subset of Yandex's catalog (Qwen3 235B, DeepSeek 3.2, GPT OSS 120B/20B,
Qwen3.6-35B, ...) is exposed only through Yandex's OpenAI-compatible HTTP
endpoint at `/v1/chat/completions`, NOT through the gRPC `YCloudML` SDK.
Trying those models via `YandexAgent` returns
`Model is not available via gRPC API. Please use HTTP OpenAI API instead.`

This backend wraps the official `openai` Python SDK with two tweaks:
  - `base_url='https://llm.api.cloud.yandex.net/v1'`
  - `Authorization: Api-Key <YANDEX_API_KEY>` (NOT `Bearer ...`)
The model identifier passed to `chat.completions.create` is the full Yandex
URI `gpt://<folder>/<short-name>` (e.g. `gpt://b1g.../qwen3-235b-a22b-fp8`).

Tool calling is the standard OpenAI shape; the ReAct driver and tool
registry plug in unchanged.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .llm import Proposal
from .react import AssistantMessage, Backend, Message, ReActDriver, ToolCall
from .tools import ToolContext, as_openai_tools, build_tool_registry


# OpenAI SDK raises `openai.APIConnectionError` / `openai.APITimeoutError`
# wrapping httpx errors. Treat both names + the underlying httpx types as
# transient so we retry rather than aborting an iteration.
_TRANSIENT_ERR_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError",
    "ReadError", "ReadTimeout", "ConnectError", "ConnectTimeout",
    "RemoteProtocolError", "TimeoutError",
}


def _is_transient(exc: BaseException) -> bool:
    if type(exc).__name__ in _TRANSIENT_ERR_NAMES:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_transient(cause)
    return False


class _YandexOpenAIBackend:
    """Thin OpenAI-SDK adapter speaking Yandex's OpenAI-compatible endpoint."""

    def __init__(self, client, model: str, tools_wire: list[dict],
                 max_retries: int = 4, base_backoff_sec: float = 2.0,
                 on_usage: Optional[Any] = None) -> None:
        self._client = client
        self._model = model
        self._tools = tools_wire
        self._max_retries = max_retries
        self._base_backoff = base_backoff_sec
        self._on_usage = on_usage

    def invoke(self, messages: list[Message]) -> AssistantMessage:
        oai_msgs = [_to_openai(m) for m in messages]
        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=oai_msgs,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=0.2,
                )
                self._record_usage(resp)
                return _from_openai(resp)
            except Exception as e:  # noqa: BLE001
                if not _is_transient(e) or attempt >= self._max_retries:
                    raise
                last_exc = e
                time.sleep(self._base_backoff * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("backend invoke loop terminated unexpectedly")

    def _record_usage(self, resp) -> None:
        if self._on_usage is None:
            return
        u = getattr(resp, "usage", None)
        if not u:
            return
        try:
            self._on_usage({
                "input": int(getattr(u, "prompt_tokens", 0) or 0),
                "output": int(getattr(u, "completion_tokens", 0) or 0),
                "total": int(getattr(u, "total_tokens", 0) or 0),
            })
        except Exception:
            pass


def _to_openai(m: Message) -> dict:
    """Translate our normalized Message into an OpenAI ChatCompletion message."""
    if m.role == "system":
        return {"role": "system", "content": m.content or " "}
    if m.role == "user":
        return {"role": "user", "content": m.content or " "}
    if m.role == "tool":
        return {"role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": m.content or " "}
    # assistant
    out: dict = {"role": "assistant", "content": m.content or None}
    if m.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name,
                          "arguments": json.dumps(tc.args, ensure_ascii=False)}}
            for tc in m.tool_calls
        ]
        # OpenAI accepts content=None for assistant turns that are pure tool_calls
    else:
        out["content"] = m.content or ""
    return out


def _from_openai(resp) -> AssistantMessage:
    choice = resp.choices[0]
    msg = choice.message
    content = (getattr(msg, "content", None) or "")
    calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        tid = getattr(tc, "id", "") or ""
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        args_raw = getattr(fn, "arguments", None) if fn else None
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {"_raw": args_raw}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        if name:
            calls.append(ToolCall(id=tid, name=name, args=args))
    return AssistantMessage(content=content, tool_calls=calls, raw=resp)


# Known Yandex AI Studio short names. The agent accepts both bare short names
# (we'll prepend `gpt://<folder>/`) and pre-formed URIs (`gpt://...`) so callers
# can pass either.
_OPENAI_SHORTNAMES = {
    "qwen3-235b-a22b-fp8", "qwen3.6-35b", "qwen3.5-35b",
    "deepseek-v3.2", "deepseek-3.2",
    "gpt-oss-120b", "gpt-oss-20b",
    "alice-ai-llm",
}


class YandexOpenAIAgent:
    """ReAct backend for Yandex AI Studio models served via OpenAI-compat HTTP."""

    def __init__(
        self,
        model: str = "qwen3-235b-a22b-fp8",
        max_tool_calls: int = 15,
        max_wallclock_sec: float = 900.0,
        temperature: float = 0.2,
        folder_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://llm.api.cloud.yandex.net/v1",
        verify_ssl: bool = True,
    ) -> None:
        try:
            import httpx
            from openai import OpenAI
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "openai SDK not installed. Install via: pip install openai"
            ) from e
        folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        api_key = (api_key or os.environ.get("YANDEX_API_KEY")
                   or os.environ.get("YC_API_KEY"))
        if not folder_id or not api_key:
            raise RuntimeError(
                "Set YANDEX_FOLDER_ID and YANDEX_API_KEY env vars (or pass "
                "folder_id=/api_key= explicitly).")
        # Yandex authenticates with `Api-Key <key>`, not OpenAI's default
        # `Bearer <key>` — override via default_headers. `api_key=` still
        # must be non-empty so OpenAI's client constructor accepts it.
        # `verify_ssl=False` is the escape hatch for hosts where corporate
        # TLS interception (e.g. Avast) produces certs that fail strict
        # OpenSSL validation; the dev does not own the trust chain.
        self._client = OpenAI(
            base_url=base_url,
            api_key="ignored",
            default_headers={
                "Authorization": f"Api-Key {api_key}",
                "x-folder-id": folder_id,
            },
            http_client=httpx.Client(verify=verify_ssl, timeout=180.0),
        )
        self._model_uri = (model if model.startswith("gpt://")
                           else f"gpt://{folder_id}/{model}")
        self._max_tool_calls = max_tool_calls
        self._max_wallclock_sec = max_wallclock_sec
        self._temperature = temperature

    def propose(self, system_prompt: str, iteration_prompt: str,
                iteration: int, context: Optional[Any] = None) -> Proposal:
        if context is None:
            raise RuntimeError("YandexOpenAIAgent requires a ToolContext via "
                               "`context=`. Run it through ResearchLoop.")
        tools = build_tool_registry()
        wire = as_openai_tools(tools)
        on_usage = lambda u: context.on_event({
            "event": "TOKEN_USAGE", "iter": context.iteration,
            "input": u["input"], "output": u["output"], "total": u["total"],
        })
        driver = ReActDriver(
            backend=_YandexOpenAIBackend(self._client, self._model_uri, wire,
                                          on_usage=on_usage),
            tools=tools, context=context,
            max_tool_calls=self._max_tool_calls,
            max_wallclock_sec=self._max_wallclock_sec,
        )
        return driver.run(system_prompt, iteration_prompt, iteration)
