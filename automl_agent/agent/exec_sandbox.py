"""
Per-iteration Python REPL the ReAct agent uses for real EDA.

The agent does *real* data analysis here: residual inspection on `oof`,
ydata-profiling alerts, per-segment error breakdowns, MI probes on candidate
features, anything pandas/numpy/sklearn can express. A persistent namespace
between tool calls means follow-up turns can reference earlier results
without recomputing them. The harness still owns CV — this sandbox cannot
mutate `incumbent.py` or affect the CV score; only `submit_solution` does.

Safety: AST screen rejects forbidden imports and calls before exec; stdout
and the last expression's repr are captured and auto-summarized; a wall-clock
timeout caps each run. Not a security boundary — for untrusted code run the
whole process in a container, matching the existing `harness/sandbox.py`
stance.
"""
from __future__ import annotations

import ast
import contextlib
import io
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from harness import eda as harness_eda
from harness.sandbox import _FORBIDDEN_CALLS, _FORBIDDEN_IMPORTS

# Things that go beyond the harness blacklist for live exec (harness blocks
# them at solution.py load time; here we block them at every REPL call).
_EXTRA_FORBIDDEN_IMPORTS = {
    "os", "sys", "pathlib", "pickle", "joblib", "ctypes", "importlib",
}
_EXTRA_FORBIDDEN_CALLS: set[str] = set()

_ALL_FORBIDDEN_IMPORTS = _FORBIDDEN_IMPORTS | _EXTRA_FORBIDDEN_IMPORTS
_ALL_FORBIDDEN_CALLS = _FORBIDDEN_CALLS | _EXTRA_FORBIDDEN_CALLS


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    value_repr: str = ""     # auto-summary of the last expression's value
    error: str = ""
    elapsed_sec: float = 0.0

    def as_text(self) -> str:
        """Pack into a single string suitable for a tool-call response."""
        parts = []
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        if self.value_repr:
            parts.append(f"--- value ---\n{self.value_repr}")
        if not self.ok:
            parts.append(f"--- error ---\n{self.error}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"[elapsed {self.elapsed_sec:.2f}s]")
        return "\n".join(parts)


def static_screen(source: str) -> list[str]:
    """Reject forbidden imports/calls. Returns list of issues (empty == clean)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in _ALL_FORBIDDEN_IMPORTS:
                    issues.append(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _ALL_FORBIDDEN_IMPORTS:
                issues.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _ALL_FORBIDDEN_CALLS:
                issues.append(f"forbidden call: {node.func.id}()")
    return issues


def _render_value(value: Any, max_chars: int = 4000) -> str:
    """Auto-summarize a value for LLM consumption — keep DataFrames usable."""
    try:
        if value is None:
            return ""
        if isinstance(value, pd.DataFrame):
            head = value.head(10)
            try:
                desc = value.describe(include="all")
            except Exception:
                desc = value.describe()
            txt = (f"DataFrame shape={value.shape}\n"
                   f"head(10):\n{head.to_string()}\n\n"
                   f"describe:\n{desc.to_string()}")
        elif isinstance(value, pd.Series):
            head = value.head(20)
            txt = (f"Series name={value.name!r} len={len(value)} "
                   f"dtype={value.dtype}\n{head.to_string()}")
        elif isinstance(value, np.ndarray):
            txt = (f"ndarray shape={value.shape} dtype={value.dtype}\n"
                   f"{np.array2string(value, threshold=50, max_line_width=120)}")
        elif isinstance(value, dict):
            txt = repr({k: _short(v) for k, v in list(value.items())[:50]})
        elif isinstance(value, (list, tuple, set)):
            seq = list(value)
            if len(seq) <= 20:
                txt = repr(value)
            else:
                txt = f"{type(value).__name__} len={len(seq)}: {seq[:50]!r}"
        else:
            txt = repr(value)
    except Exception as e:
        txt = f"<repr failed: {e!r}>"
    if len(txt) > max_chars:
        txt = txt[:max_chars] + f"\n... [truncated, total {len(txt)} chars]"
    return txt


def _short(v: Any, n: int = 100) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def _split_last_expression(source: str) -> tuple[str, Optional[str]]:
    """If the last top-level statement is an expression, return (body, expr).

    Body is the source minus that final expression; expr is its source string.
    Otherwise returns (source, None).
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return source, None
    if not tree.body:
        return source, None
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return source, None
    # AST 3.8+ exposes end_lineno / end_col_offset, which we use to slice.
    lines = source.splitlines(keepends=True)
    if last.end_lineno is None:
        return source, None
    # body before the expression
    body_lines = lines[: last.lineno - 1]
    # everything on the expression's first line up to its start col
    first_expr_line_prefix = lines[last.lineno - 1][: last.col_offset]
    body = "".join(body_lines) + first_expr_line_prefix
    # expression source
    expr_lines = lines[last.lineno - 1: last.end_lineno]
    if len(expr_lines) == 1:
        expr_src = expr_lines[0][last.col_offset: last.end_col_offset]
    else:
        first = expr_lines[0][last.col_offset:]
        middle = expr_lines[1:-1]
        last_line = expr_lines[-1][: last.end_col_offset]
        expr_src = "".join([first] + middle + [last_line])
    return body, expr_src.strip()


class Sandbox:
    """Persistent-namespace Python REPL with safety screens and a wall-clock cap."""

    def __init__(
        self,
        spec,
        train: pd.DataFrame,
        test: pd.DataFrame,
        oof: Optional[np.ndarray],
        incumbent_source: str,
        feature_importance: Optional[pd.Series],
        workdir: Path,
        timeout_sec: int = 10,
        max_stdout: int = 4000,
        max_stderr: int = 2000,
        max_value_chars: int = 4000,
    ) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.timeout_sec = timeout_sec
        self.max_stdout = max_stdout
        self.max_stderr = max_stderr
        self.max_value_chars = max_value_chars

        # Modules / objects exposed to the agent. `.copy()` so user mutations
        # do not leak back into harness state.
        self._initial_ns: dict[str, Any] = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "train": train.copy(),
            "test": test.copy(),
            "spec": spec,
            "oof": (None if oof is None else np.array(oof, copy=True)),
            "incumbent_source": incumbent_source,
            "feature_importance": (None if feature_importance is None
                                   else feature_importance.copy()),
            "eda": harness_eda,
        }
        try:
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
            self._initial_ns["scipy"] = __import__("scipy")
            self._initial_ns["sklearn"] = __import__("sklearn")
        except Exception:
            pass
        self.ns: dict[str, Any] = dict(self._initial_ns)
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="sandbox")

    def reset(self) -> None:
        self.ns = dict(self._initial_ns)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- main entry point -------------------------------------------------- #
    def run(self, code: str) -> ExecResult:
        t0 = time.time()
        issues = static_screen(code)
        if issues:
            return ExecResult(ok=False,
                              error="rejected: " + "; ".join(issues),
                              elapsed_sec=time.time() - t0)

        body, expr_src = _split_last_expression(code)

        try:
            future = self._executor.submit(self._exec, body, expr_src)
            stdout, stderr, value, err = future.result(
                timeout=self.timeout_sec)
        except FutTimeout:
            # NOTE: we cannot truly kill the running thread in CPython. The
            # caller's wall-clock is bounded; if a runaway pandas op holds
            # the GIL longer than `timeout_sec`, this call returns and the
            # rogue thread keeps running until it yields. Best-effort.
            return ExecResult(ok=False,
                              error=f"timeout after {self.timeout_sec}s",
                              elapsed_sec=time.time() - t0)
        except Exception as e:  # noqa: BLE001
            return ExecResult(ok=False, error=f"executor failure: {e!r}",
                              elapsed_sec=time.time() - t0)

        return ExecResult(
            ok=(err is None),
            stdout=stdout[: self.max_stdout],
            stderr=stderr[: self.max_stderr],
            value_repr=_render_value(value, self.max_value_chars),
            error=(err or ""),
            elapsed_sec=time.time() - t0,
        )

    def _exec(self, body: str, expr_src: Optional[str]):
        out, err_buf = io.StringIO(), io.StringIO()
        value = None
        err_msg: Optional[str] = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err_buf):
                if body.strip():
                    compiled = compile(body, "<sandbox>", "exec")
                    exec(compiled, self.ns, self.ns)
                if expr_src is not None:
                    compiled_expr = compile(expr_src, "<sandbox-expr>", "eval")
                    value = eval(compiled_expr, self.ns, self.ns)
        except Exception:
            err_msg = traceback.format_exc(limit=8)
        return out.getvalue(), err_buf.getvalue(), value, err_msg
