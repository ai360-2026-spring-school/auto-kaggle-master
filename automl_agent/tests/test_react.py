"""Tests for the backend-agnostic ReAct driver."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.exec_sandbox import Sandbox
from agent.react import AssistantMessage, Message, ReActDriver, ToolCall
from agent.tools import ToolContext, build_tool_registry


class ScriptedBackend:
    """A Backend that emits a fixed list of pre-canned AssistantMessages."""
    def __init__(self, script: list[AssistantMessage]):
        self.script = list(script)
        self.calls: list[list[Message]] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.script:
            return AssistantMessage(content="(no more scripted responses)")
        return self.script.pop(0)


_VALID_SOLUTION = (
    "from harness import BaseSolution\n"
    "import pandas as pd\n"
    "class Solution(BaseSolution):\n"
    "    def fit(self, train_df, y, spec): pass\n"
    "    def transform(self, df):\n"
    "        return df.select_dtypes(include='number')\n"
)


@pytest.fixture
def ctx(tmp_path):
    train = pd.DataFrame({"id": range(10), "x": np.arange(10.0),
                          "y": np.arange(10) * 0.5})
    spec = SimpleNamespace(train=train, test=train.copy(),
                           target_col="y", id_col="id",
                           problem_type="regression", n_classes=1,
                           class_labels=None)
    sb = Sandbox(spec=spec, train=train, test=train,
                 oof=np.zeros(10), incumbent_source="class Solution: pass",
                 feature_importance=None, workdir=tmp_path, timeout_sec=3)
    inc = tmp_path / "incumbent.py"
    inc.write_text("class Solution: pass\n")
    jp = tmp_path / "journal.jsonl"
    jp.touch()
    nb = tmp_path / "eda_notebook.md"
    events = []
    yield ToolContext(
        sandbox=sb, workdir=tmp_path, incumbent_path=inc,
        journal_path=jp, notebook_path=nb,
        incumbent_score=0.0, metric_name="r2",
        on_event=events.append,
    ), events
    sb.close()


def test_react_terminates_on_submit_solution(ctx):
    ctx_, events = ctx
    backend = ScriptedBackend([
        AssistantMessage(
            content="Let me explore first.",
            tool_calls=[ToolCall(id="t1", name="python_exec",
                                 args={"code": "train.shape"})],
        ),
        AssistantMessage(
            content="Submitting now.",
            tool_calls=[ToolCall(id="t2", name="submit_solution",
                                 args={"code": _VALID_SOLUTION,
                                       "hypothesis": "use numeric cols only",
                                       "expected_effect": "+0.01 R2"})],
        ),
    ])
    driver = ReActDriver(backend=backend, tools=build_tool_registry(),
                         context=ctx_, max_tool_calls=10)
    prop = driver.run("system", "user", iteration=0)
    assert "class Solution" in prop.solution_source
    assert prop.hypothesis == "use numeric cols only"
    assert prop.expected_effect == "+0.01 R2"
    assert len(prop.tool_trace) == 2
    assert {e["event"] for e in events} >= {"TOOL_CALL", "TOOL_RESULT"}


def test_react_extracts_code_from_text_when_no_tool_call(ctx):
    ctx_, _ = ctx
    text = (
        "Here is my solution:\n\n```python\n"
        + _VALID_SOLUTION
        + "```\n"
    )
    backend = ScriptedBackend([AssistantMessage(content=text)])
    driver = ReActDriver(backend=backend, tools=build_tool_registry(),
                         context=ctx_, max_tool_calls=5)
    # The driver will first nudge ("you did not call any tool"), so we provide
    # a second response: still the same text with the code block.
    backend.script.append(AssistantMessage(content=text))
    prop = driver.run("system", "user", iteration=0)
    assert "class Solution" in prop.solution_source


def test_react_tool_budget_exhausted_falls_through(ctx):
    ctx_, _ = ctx
    # Exactly max_tool_calls explore turns, then a 4th turn that would push
    # beyond the budget -> driver hits final_attempt -> backend responds to
    # the nudge with submit_solution.
    script = [
        AssistantMessage(
            content="exploring",
            tool_calls=[ToolCall(id=f"t{i}", name="python_exec",
                                 args={"code": "1"})],
        ) for i in range(3)
    ]
    script.append(AssistantMessage(
        content="one more probe",
        tool_calls=[ToolCall(id="t3", name="python_exec",
                             args={"code": "2"})],
    ))
    script.append(AssistantMessage(
        content="ok submitting",
        tool_calls=[ToolCall(id="tfinal", name="submit_solution",
                             args={"code": _VALID_SOLUTION,
                                   "hypothesis": "fallback"})],
    ))
    backend = ScriptedBackend(script)
    driver = ReActDriver(backend=backend, tools=build_tool_registry(),
                         context=ctx_, max_tool_calls=3)
    prop = driver.run("system", "user", iteration=0)
    assert "class Solution" in prop.solution_source
    assert prop.hypothesis == "fallback"


def test_react_raises_if_no_solution_ever(ctx):
    ctx_, _ = ctx
    backend = ScriptedBackend([
        AssistantMessage(content="I refuse"),
        AssistantMessage(content="still refusing"),  # the nudge response
    ])
    driver = ReActDriver(backend=backend, tools=build_tool_registry(),
                         context=ctx_, max_tool_calls=5)
    with pytest.raises(RuntimeError):
        driver.run("system", "user", iteration=0)
