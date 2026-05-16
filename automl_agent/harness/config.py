"""
Fixed configuration for the harness.

This file is the equivalent of karpathy/autoresearch's `prepare.py` constants:
it is part of the *locked* infrastructure. The agent never edits this. It
defines the budgets that make experiments comparable and the CatBoost config
that is frozen across the whole run so that the only thing that moves the
metric is the agent's data work.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --- The LOCKED model. The agent may not change a single value here. ---------
# A single, universal config that works acceptably on almost any tabular task.
# It is deliberately *not* tuned: tuning is not the agent's job, data is.
LOCKED_CATBOST_PARAMS: dict = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "leaf_estimation_iterations": 1,
    "od_type": "Iter",
    "od_wait": 150,            # early stopping patience on the fold's eval set
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": False,
}


@dataclass
class Budget:
    """Budgets that keep the autonomous loop bounded and experiments fair."""
    max_iterations: int = 25          # agent edit/evaluate cycles
    max_seconds_total: float = 60 * 60
    max_seconds_per_eval: float = 60 * 15
    cv_folds: int = 5
    cv_repeats: int = 1
    # An experiment must beat the incumbent by at least this (relative to the
    # metric's std across folds) to be accepted. Prevents chasing CV noise.
    min_improvement_sigmas: float = 0.15


@dataclass
class Paths:
    workdir: str = "runs/latest"
    solution_file: str = "solution.py"        # the file the agent rewrites
    incumbent_file: str = "incumbent.py"      # best solution so far
    journal_file: str = "journal.jsonl"       # every experiment, like a lab log
    submission_file: str = "submission.csv"


@dataclass
class HarnessConfig:
    budget: Budget = field(default_factory=Budget)
    paths: Paths = field(default_factory=Paths)
    catboost_params: dict = field(default_factory=lambda: dict(LOCKED_CATBOST_PARAMS))
    random_seed: int = 42
