"""
Backend-agnostic ReAct driver.

Owns the multi-turn message history for one research iteration: alternates
between asking the model for the next action (via `Backend.invoke`) and
running any tool calls the model emits against `ToolContext`. Terminates
when:
  - the model fires `submit_solution` (success path),
  - the tool budget is exhausted (driver gives the model one last chance),
  - or the model emits an answer with no tool calls (degenerate path —
    we try to extract a code block from the text as a fallback Proposal).

Decoupled from any specific SDK. Each backend (GigaChat, Yandex AI Studio)
implements a tiny `Backend` adapter that turns a provider-agnostic `Message`
list into a provider-native API call and parses the response back into a
normalized `AssistantMessage`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .llm import Proposal
from .tools import ToolContext, ToolResult, ToolSpec, dispatch


# --------------------------------------------------------------------------- #
#  Normalized message types                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Message:
    role: str                              # 'system' | 'user' | 'assistant' | 'tool'
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None     # set for role='tool'
    name: Optional[str] = None             # tool name when role='tool'
    # Provider-native object for assistant turns that contained tool_calls.
    # Yandex's protocol REQUIRES the original ToolCallList object to be
    # re-fed when echoing the assistant turn back to the model — a plain
    # rendered text won't satisfy the "prior tool call" check on tool_results.
    raw_tool_calls: Optional[Any] = None


@dataclass
class AssistantMessage:
    """What a Backend returns from invoke()."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None                        # provider-specific object (for debugging)
    raw_tool_calls: Optional[Any] = None   # provider-native ToolCallList (e.g. Yandex)


class Backend(Protocol):
    def invoke(self, messages: list[Message]) -> AssistantMessage: ...


# Proposal lives in agent.llm (single source of truth) — imported above.


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def _extract_code(text: str) -> Optional[str]:
    m = _CODE_RE.search(text or "")
    return m.group(1).strip() if m else None


def _short(v: Any, n: int = 240) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #


class ReActDriver:
    def __init__(
        self,
        backend: Backend,
        tools: list[ToolSpec],
        context: ToolContext,
        max_tool_calls: int = 15,
        max_wallclock_sec: float = 900.0,
    ) -> None:
        self._backend = backend
        self._tools = tools
        self._ctx = context
        self._max_tool_calls = max_tool_calls
        self._max_wallclock_sec = max_wallclock_sec

    def run(self, system_prompt: str, iteration_prompt: str,
            iteration: int) -> Proposal:
        self._ctx.iteration = iteration
        history: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=iteration_prompt),
        ]
        trace: list[dict] = []
        t0 = time.time()
        tool_calls_used = 0
        nudged_no_tool = False

        while True:
            # wall-clock guard
            if time.time() - t0 > self._max_wallclock_sec:
                self._ctx.on_event({
                    "event": "REACT_TIMEOUT",
                    "iter": iteration,
                    "msg": f"wallclock {self._max_wallclock_sec}s exceeded",
                })
                return self._final_attempt(history, trace,
                                           "wallclock budget exhausted")

            ai = self._backend.invoke(history)
            history.append(Message(role="assistant", content=ai.content,
                                   tool_calls=ai.tool_calls,
                                   raw_tool_calls=ai.raw_tool_calls))

            # No tool calls: model is talking plain text. Either it's done
            # reasoning (we extract code from the message as a Proposal), or
            # it stalled. Either way, stop.
            if not ai.tool_calls:
                if self._ctx.submitted:
                    return self._submitted_proposal(ai.content, trace)
                # try to salvage a code block from the assistant message
                code = _extract_code(ai.content)
                if code and "class Solution" in code:
                    return Proposal(
                        reasoning=ai.content[:2000],
                        solution_source=code,
                        hypothesis="(no submit_solution call; extracted "
                                   "from assistant text)",
                        tool_trace=trace,
                    )
                # Give exactly one nudge to call submit_solution.
                if not nudged_no_tool:
                    nudged_no_tool = True
                    history.append(Message(
                        role="user",
                        content=("You did not call any tool. Call "
                                 "submit_solution with a complete "
                                 "solution.py NOW."),
                    ))
                    continue
                return self._final_attempt(history, trace,
                                           "model returned no tool call")

            # Dispatch every tool call this turn.
            for tc in ai.tool_calls:
                if tool_calls_used >= self._max_tool_calls:
                    history.append(Message(
                        role="tool",
                        content=("Tool budget exhausted before this call "
                                 "could run."),
                        tool_call_id=tc.id, name=tc.name,
                    ))
                    return self._final_attempt(history, trace,
                                               "tool budget exhausted")
                tool_calls_used += 1
                args_preview = _short(tc.args)
                self._ctx.on_event({
                    "event": "TOOL_CALL", "iter": iteration,
                    "name": tc.name, "args_preview": args_preview,
                })
                result = dispatch(tc.name, tc.args, self._tools, self._ctx)
                trace.append({"name": tc.name, "args_preview": args_preview,
                              "ok": not result.is_error,
                              "content_preview": _short(result.content, 400)})
                self._ctx.on_event({
                    "event": "TOOL_RESULT", "iter": iteration,
                    "name": tc.name, "ok": not result.is_error,
                    "content_len": len(result.content),
                })
                history.append(Message(role="tool", content=result.content,
                                       tool_call_id=tc.id, name=tc.name))

                if self._ctx.submitted is not None:
                    return self._submitted_proposal(ai.content, trace)

    # -- finalization paths ------------------------------------------------ #

    def _submitted_proposal(self, last_text: str,
                            trace: list[dict]) -> Proposal:
        s = self._ctx.submitted
        assert s is not None
        return Proposal(
            reasoning=(s.hypothesis or (last_text or "")[:2000]),
            solution_source=s.code,
            hypothesis=s.hypothesis,
            expected_effect=s.expected_effect,
            tool_trace=trace,
        )

    def _final_attempt(self, history: list[Message], trace: list[dict],
                       reason: str) -> Proposal:
        """Last-chance turn: nudge the model to submit; if it doesn't, raise."""
        history.append(Message(
            role="user",
            content=(f"You must end this iteration. Reason: {reason}. "
                     "Call submit_solution exactly once with a full "
                     "solution.py. Do not run any other tool."),
        ))
        try:
            ai = self._backend.invoke(history)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"backend failed during final attempt: {e!r}") from e
        if ai.tool_calls:
            for tc in ai.tool_calls:
                if tc.name != "submit_solution":
                    continue
                result = dispatch(tc.name, tc.args, self._tools, self._ctx)
                if self._ctx.submitted is not None:
                    return self._submitted_proposal(ai.content, trace)
        code = _extract_code(ai.content)
        if code and "class Solution" in code:
            return Proposal(
                reasoning=ai.content[:2000], solution_source=code,
                hypothesis="(salvaged from final-attempt text)",
                tool_trace=trace,
            )
        raise RuntimeError(
            f"agent failed to submit a solution ({reason}). "
            "Last assistant message: " + _short(ai.content, 400))
