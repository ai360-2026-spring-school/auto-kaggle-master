"""Tests for agent.tools — ToolSpec wiring, schemas, dispatch behavior."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent.exec_sandbox import Sandbox
from agent.tools import (ToolContext, as_anthropic_tools, as_gigachat_tools,
                         as_openai_tools, as_yandex_tools, build_tool_registry,
                         dispatch)


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
                 feature_importance=pd.Series({"x": 1.0}),
                 workdir=tmp_path, timeout_sec=3)
    inc = tmp_path / "incumbent.py"
    inc.write_text("class Solution: pass\n")
    jp = tmp_path / "journal.jsonl"
    jp.write_text(json.dumps({"event": "BASELINE", "score": 0.5}) + "\n")
    nb = tmp_path / "eda_notebook.md"
    events = []
    yield ToolContext(
        sandbox=sb, workdir=tmp_path, incumbent_path=inc,
        journal_path=jp, notebook_path=nb,
        incumbent_score=0.5, metric_name="r2",
        on_event=events.append,
    )
    sb.close()


def test_registry_complete():
    names = {t.name for t in build_tool_registry()}
    assert names == {"python_exec", "read_incumbent", "read_journal",
                     "add_insight", "submit_solution"}


def test_openai_schema_shape():
    schemas = as_openai_tools(build_tool_registry())
    assert all(s["type"] == "function" for s in schemas)
    assert all("function" in s and "parameters" in s["function"]
               for s in schemas)


def test_anthropic_schema_shape():
    schemas = as_anthropic_tools(build_tool_registry())
    assert all("input_schema" in s for s in schemas)


def test_aliases_match():
    base = as_openai_tools(build_tool_registry())
    assert as_gigachat_tools(build_tool_registry()) == base
    assert as_yandex_tools(build_tool_registry()) == base


def test_python_exec_dispatch(ctx):
    r = dispatch("python_exec", {"code": "1 + 1"},
                 build_tool_registry(), ctx)
    assert not r.is_error
    assert "2" in r.content


def test_python_exec_empty_code(ctx):
    r = dispatch("python_exec", {"code": ""},
                 build_tool_registry(), ctx)
    assert r.is_error


def test_read_incumbent(ctx):
    r = dispatch("read_incumbent", {}, build_tool_registry(), ctx)
    assert not r.is_error
    assert "class Solution" in r.content


def test_read_journal_filters_tool_events(ctx):
    # add some noise the tool should filter out
    with ctx.journal_path.open("a") as f:
        f.write(json.dumps({"event": "TOOL_CALL", "name": "python_exec"}) + "\n")
        f.write(json.dumps({"event": "RESULT", "score": 0.6}) + "\n")
    r = dispatch("read_journal", {"last_n": 10}, build_tool_registry(), ctx)
    assert not r.is_error
    assert "TOOL_CALL" not in r.content
    assert "BASELINE" in r.content
    assert "RESULT" in r.content


def test_add_insight_persists(ctx):
    r = dispatch("add_insight", {"text": "high cardinality on Driver col"},
                 build_tool_registry(), ctx)
    assert not r.is_error
    assert ctx.notebook_path.exists()
    assert "high cardinality" in ctx.notebook_path.read_text(encoding="utf-8")


def test_submit_solution_rejects_no_class(ctx):
    r = dispatch("submit_solution",
                 {"code": "import pandas", "hypothesis": "x"},
                 build_tool_registry(), ctx)
    assert r.is_error
    assert "class Solution" in r.content
    assert ctx.submitted is None


def test_submit_solution_rejects_forbidden(ctx):
    code = "import catboost\nclass Solution: pass"
    r = dispatch("submit_solution",
                 {"code": code, "hypothesis": "h"},
                 build_tool_registry(), ctx)
    assert r.is_error
    assert "catboost" in r.content
    assert ctx.submitted is None


def test_submit_solution_accepts_valid(ctx):
    code = (
        "from harness import BaseSolution\n"
        "import pandas as pd\n"
        "class Solution(BaseSolution):\n"
        "    def fit(self, train_df, y, spec): pass\n"
        "    def transform(self, df): return df.select_dtypes(include='number')\n"
    )
    r = dispatch("submit_solution",
                 {"code": code, "hypothesis": "h", "expected_effect": "+0.01"},
                 build_tool_registry(), ctx)
    assert not r.is_error
    assert ctx.submitted is not None
    assert ctx.submitted.hypothesis == "h"


def test_dispatch_unknown_tool(ctx):
    r = dispatch("nope", {}, build_tool_registry(), ctx)
    assert r.is_error
    assert "unknown" in r.content


def test_submit_solution_strips_markdown_fences(ctx):
    raw = (
        "from harness import BaseSolution\n"
        "import pandas as pd\n"
        "class Solution(BaseSolution):\n"
        "    def fit(self, train_df, y, spec): pass\n"
        "    def transform(self, df): return df.select_dtypes(include='number')\n"
    )
    wrapped = "```python\n" + raw + "```"
    r = dispatch("submit_solution",
                 {"code": wrapped, "hypothesis": "h"},
                 build_tool_registry(), ctx)
    assert not r.is_error, r.content
    assert ctx.submitted is not None
    assert ctx.submitted.code.rstrip() == raw.rstrip()
    assert "```" not in ctx.submitted.code


def test_python_exec_strips_markdown_fences(ctx):
    r = dispatch("python_exec", {"code": "```python\n1 + 2\n```"},
                 build_tool_registry(), ctx)
    assert not r.is_error, r.content
    assert "3" in r.content


def test_submit_solution_decodes_literal_escapes(ctx):
    """GigaChat sometimes sends `code` with literal '\\n' rather than newlines."""
    raw = (
        "from harness import BaseSolution\n"
        "import pandas as pd\n"
        "class Solution(BaseSolution):\n"
        "    def fit(self, train_df, y, spec): pass\n"
        "    def transform(self, df): return df.select_dtypes(include='number')\n"
    )
    wrapped = "```python\n" + raw + "```"
    literal_escaped = wrapped.replace("\n", "\\n")
    assert "\n" not in literal_escaped and "\\n" in literal_escaped
    r = dispatch("submit_solution",
                 {"code": literal_escaped, "hypothesis": "h"},
                 build_tool_registry(), ctx)
    assert not r.is_error, r.content
    assert ctx.submitted is not None
    assert "class Solution" in ctx.submitted.code
    assert "\\n" not in ctx.submitted.code


def test_decode_leaves_valid_python_alone(ctx):
    """If code is already valid Python (with real newlines AND a literal
    \\n inside a docstring), don't touch it."""
    code = 'def f():\n    """has \\n inside docstring"""\n    return 1\nf()'
    r = dispatch("python_exec", {"code": code},
                 build_tool_registry(), ctx)
    assert not r.is_error, r.content
    assert "1" in r.content


def test_python_exec_decodes_mixed_escapes(ctx):
    """Code with mostly literal \\n that does NOT parse as-is should be
    decoded and run successfully."""
    fully_escaped = "import pandas as pd\\nx = train.shape\\nx"
    r = dispatch("python_exec", {"code": fully_escaped},
                 build_tool_registry(), ctx)
    assert not r.is_error, r.content
    assert "(10," in r.content


def test_submit_solution_strips_unclosed_leading_fence(ctx):
    """GigaChat sometimes emits only the opening ```python with no closing
    fence; we should still extract the code below it."""
    raw = (
        "from harness import BaseSolution\n"
        "import pandas as pd\n"
        "class Solution(BaseSolution):\n"
        "    def fit(self, train_df, y, spec): pass\n"
        "    def transform(self, df): return df.select_dtypes(include='number')\n"
    )
    open_only = "```python\n" + raw
    r = dispatch("submit_solution",
                 {"code": open_only, "hypothesis": "h"},
                 build_tool_registry(), ctx)
    assert not r.is_error, r.content
    assert ctx.submitted is not None
    assert ctx.submitted.code.startswith("from harness")
