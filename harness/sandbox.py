"""
Sandbox for loading agent-authored solution.py.

This isolates the *import* of agent code and statically screens it for the
obviously dangerous stuff (network, subprocess, fs escapes, and — importantly
for this design — importing catboost directly, which would let the agent
sidestep the locked model). It is a guardrail, not a security boundary; for
untrusted operation run the whole process in a container/seccomp jail. That
matches autoresearch's stance: "disable all permissions" is done at the
sandbox/container level, not inside the prompt.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from .contract import BaseSolution

_FORBIDDEN_IMPORTS = {
    "catboost", "xgboost", "lightgbm", "subprocess", "socket",
    "requests", "urllib", "shutil", "multiprocessing",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open"}


def static_check(source: str) -> list[str]:
    """Return a list of policy violations (empty == clean)."""
    issues: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    issues.append(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                issues.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                issues.append(f"forbidden call: {node.func.id}()")
    return issues


def load_solution(path: str | Path) -> BaseSolution:
    """Import solution.py and instantiate its Solution class."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    issues = static_check(source)
    if issues:
        raise PermissionError("solution.py rejected: " + "; ".join(issues))

    spec = importlib.util.spec_from_file_location("agent_solution", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_solution"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if not hasattr(mod, "Solution"):
        raise AttributeError("solution.py must define a class named `Solution`.")
    obj = mod.Solution()
    if not isinstance(obj, BaseSolution):
        raise TypeError("Solution must subclass harness.contract.BaseSolution.")
    return obj
