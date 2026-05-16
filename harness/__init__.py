"""Locked harness. The agent imports nothing from here except BaseSolution."""
from .config import HarnessConfig, Budget, Paths, LOCKED_CATBOST_PARAMS
from .contract import BaseSolution
from .data import build_problem_spec, ProblemSpec
from .metrics import select_metric, Metric
from .cv import evaluate_solution, is_improvement, EvalResult
from .sandbox import load_solution
from . import eda

__all__ = [
    "HarnessConfig", "Budget", "Paths", "LOCKED_CATBOST_PARAMS",
    "BaseSolution", "build_problem_spec", "ProblemSpec",
    "select_metric", "Metric", "evaluate_solution", "is_improvement",
    "EvalResult", "load_solution", "eda",
]
