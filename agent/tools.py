"""
Tool registry the ReAct agent calls during one research iteration.

One declarative source (`ToolSpec`) is converted to each backend's wire
format (`as_openai_tools`, `as_anthropic_tools`, `as_gigachat_tools`,
`as_yandex_tools`). The handlers run in-process and share a `ToolContext`
that bundles the sandbox + relevant filesystem paths.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from harness.sandbox import static_check as _solution_static_check

from .exec_sandbox import Sandbox


# LLMs frequently wrap code arguments in markdown code fences even though the
# tool schema specifies a plain string. Strip a single outer fence so
# `submit_solution` and `python_exec` actually parse.
_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)\n```\s*$", re.S)


def _strip_code_fences(code: str) -> str:
    if not isinstance(code, str):
        return code
    m = _FENCE_RE.match(code)
    if m:
        return m.group(1)
    return code

# --------------------------------------------------------------------------- #
#  Dataclasses                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class SubmittedSolution:
    code: str
    hypothesis: str
    expected_effect: str


@dataclass
class ToolContext:
    sandbox: Sandbox
    workdir: Path
    incumbent_path: Path
    journal_path: Path
    notebook_path: Path
    incumbent_score: Optional[float]
    metric_name: str
    on_event: Callable[[dict], None]                # ResearchLoop._log
    iteration: int = 0
    submitted: Optional[SubmittedSolution] = None   # set by submit_solution


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict                # JSON-Schema, OpenAI/Anthropic-style
    handler: Callable[[dict, ToolContext], ToolResult]


# --------------------------------------------------------------------------- #
#  Handlers                                                                   #
# --------------------------------------------------------------------------- #


def _h_python_exec(args: dict, ctx: ToolContext) -> ToolResult:
    code = _strip_code_fences(args.get("code", ""))
    if not isinstance(code, str) or not code.strip():
        return ToolResult("error: `code` must be a non-empty string.",
                          is_error=True)
    res = ctx.sandbox.run(code)
    return ToolResult(res.as_text(), is_error=not res.ok)


def _h_read_incumbent(_args: dict, ctx: ToolContext) -> ToolResult:
    try:
        txt = ctx.incumbent_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult("incumbent not found yet.", is_error=True)
    return ToolResult(f"--- incumbent.py ---\n{txt}")


_NOISY_EVENTS = {"TOOL_CALL", "TOOL_RESULT"}


def _h_read_journal(args: dict, ctx: ToolContext) -> ToolResult:
    last_n = int(args.get("last_n", 5))
    if not ctx.journal_path.exists():
        return ToolResult("(journal is empty)")
    lines = [l for l in ctx.journal_path.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    filtered = []
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") in _NOISY_EVENTS:
            continue
        filtered.append(rec)
    tail = filtered[-last_n:]
    return ToolResult("\n".join(json.dumps(r, default=str) for r in tail)
                      or "(journal has no non-tool events yet)")


def _h_add_insight(args: dict, ctx: ToolContext) -> ToolResult:
    from . import notebook
    text = args.get("text", "").strip()
    if not text:
        return ToolResult("error: `text` must be non-empty.", is_error=True)
    notebook.append_insight(ctx.notebook_path, text, iteration=ctx.iteration)
    return ToolResult(f"insight saved (len={len(text)} chars).")


def _h_submit_solution(args: dict, ctx: ToolContext) -> ToolResult:
    code = _strip_code_fences(args.get("code", ""))
    hypothesis = args.get("hypothesis", "").strip()
    expected = args.get("expected_effect", "").strip()
    if not isinstance(code, str) or "class Solution" not in code:
        return ToolResult("rejected: `code` must define `class Solution`. "
                          "Send raw Python source, NOT wrapped in markdown "
                          "code fences.", is_error=True)
    issues = _solution_static_check(code)
    if issues:
        return ToolResult(
            "rejected by static_check (fix and resubmit): "
            + "; ".join(issues),
            is_error=True,
        )
    ctx.submitted = SubmittedSolution(code=code, hypothesis=hypothesis,
                                      expected_effect=expected)
    return ToolResult(
        f"solution accepted for CV evaluation "
        f"(hypothesis={hypothesis[:120]!r}). The ReAct loop will now close "
        f"and the harness will score it.",
    )


# --------------------------------------------------------------------------- #
#  Registry                                                                   #
# --------------------------------------------------------------------------- #


PYTHON_EXEC = ToolSpec(
    name="python_exec",
    description=(
        "Execute Python in a persistent sandbox. Preloaded names: `train`, "
        "`test` (pandas DataFrames, COPIES — safe to mutate), `spec` "
        "(.target_col/.id_col/.problem_type/.n_classes), `oof` (np.ndarray "
        "of OOF predictions of the current incumbent, or None), "
        "`incumbent_source` (str), `feature_importance` (pd.Series or None). "
        "Modules available: pd, np, scipy, sklearn, and `eda` "
        "(eda.profile/leakage_scan/target_relation/interaction_scan and, if "
        "ydata-profiling is installed, eda.ydata_profile). Variables persist "
        "across calls in this iteration. Per-call wall-clock cap 60s. "
        "Forbidden: catboost/xgboost/lightgbm, network, file I/O, os/sys. "
        "End your code with a bare expression to get its auto-summarized "
        "repr returned."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source to execute in the sandbox.",
            }
        },
        "required": ["code"],
    },
    handler=_h_python_exec,
)

READ_INCUMBENT = ToolSpec(
    name="read_incumbent",
    description=("Return the full source text of the current best solution.py "
                 "(the incumbent the harness keeps refining)."),
    json_schema={"type": "object", "properties": {}},
    handler=_h_read_incumbent,
)

READ_JOURNAL = ToolSpec(
    name="read_journal",
    description=("Return the tail of journal.jsonl (BASELINE/RESULT/ACCEPT/"
                 "EVAL_ERROR events; tool-call noise is filtered out)."),
    json_schema={
        "type": "object",
        "properties": {
            "last_n": {"type": "integer",
                       "description": "How many events to return (default 5).",
                       "default": 5, "minimum": 1, "maximum": 50},
        },
    },
    handler=_h_read_journal,
)

ADD_INSIGHT = ToolSpec(
    name="add_insight",
    description=("Append a short note to eda_notebook.md so future iterations "
                 "can read it. Use this for durable facts (e.g. 'col X has "
                 "75% zeros'), not for ephemeral thoughts."),
    json_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string",
                     "description": "Insight to persist (one or two sentences)."},
        },
        "required": ["text"],
    },
    handler=_h_add_insight,
)

SUBMIT_SOLUTION = ToolSpec(
    name="submit_solution",
    description=(
        "Finalize this iteration. Provide a COMPLETE new solution.py that "
        "defines `class Solution(BaseSolution)` with fit/transform/"
        "[postprocess]. The harness will CV-evaluate it and keep it only if "
        "it beats the incumbent by ~0.15 fold-std. Pre-validated by "
        "static_check; obvious violations (forbidden imports, missing "
        "Solution class) are rejected immediately so you can retry."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "code": {"type": "string",
                     "description": "Full solution.py source."},
            "hypothesis": {"type": "string",
                           "description": "One-sentence claim about why this "
                                          "should improve CV."},
            "expected_effect": {"type": "string",
                                "description": "Direction and rough magnitude "
                                               "(e.g. '+0.01 AUC')."},
        },
        "required": ["code", "hypothesis"],
    },
    handler=_h_submit_solution,
)


def build_tool_registry() -> list[ToolSpec]:
    return [PYTHON_EXEC, READ_INCUMBENT, READ_JOURNAL, ADD_INSIGHT,
            SUBMIT_SOLUTION]


# --------------------------------------------------------------------------- #
#  Per-backend wire format adapters                                           #
# --------------------------------------------------------------------------- #


def as_openai_tools(tools: list[ToolSpec]) -> list[dict]:
    """OpenAI / GigaChat (langchain) / Yandex shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            },
        }
        for t in tools
    ]


def as_anthropic_tools(tools: list[ToolSpec]) -> list[dict]:
    """Anthropic native tool-use shape."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.json_schema,
        }
        for t in tools
    ]


# Aliases for clarity at call-sites. The underlying schema is OpenAI-style and
# all three providers accept it via their langchain integrations or native SDK.
as_gigachat_tools = as_openai_tools
as_yandex_tools = as_openai_tools


def dispatch(name: str, args: dict, tools: list[ToolSpec],
             ctx: ToolContext) -> ToolResult:
    for t in tools:
        if t.name == name:
            try:
                return t.handler(args or {}, ctx)
            except Exception as e:  # noqa: BLE001 — bubble up as tool error
                return ToolResult(f"tool {name!r} raised: {e!r}", is_error=True)
    return ToolResult(f"unknown tool: {name!r}", is_error=True)
