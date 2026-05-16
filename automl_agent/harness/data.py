"""
Ingestion + automatic problem framing.

Given the four competition inputs (task text, data text, train, test), this
module figures out, *without the agent*, the things that must be fixed for the
metric to be trustworthy: which column is the target, which is the id, whether
it is classification or regression, and the natural CV scheme. The agent is
told these conclusions but cannot override the target/CV decision — that would
let it accidentally (or "helpfully") leak.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ProblemSpec:
    task_description: str
    data_description: str
    target_col: str
    id_col: Optional[str]
    problem_type: str           # 'binary' | 'multiclass' | 'regression'
    n_classes: int
    class_labels: Optional[list]
    train: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        return (
            f"problem_type={self.problem_type} n_classes={self.n_classes} "
            f"target='{self.target_col}' id='{self.id_col}' "
            f"n_train={len(self.train)} n_test={len(self.test)} "
            f"n_features={self.train.shape[1] - 1 - (1 if self.id_col else 0)}"
        )


def _guess_target(train: pd.DataFrame, test: pd.DataFrame, text: str) -> str:
    """Target = column present in train but not test, else a name hinted in text."""
    only_in_train = [c for c in train.columns if c not in test.columns]
    if len(only_in_train) == 1:
        return only_in_train[0]
    candidates = ["target", "label", "y", "class", "outcome", "prediction"]
    lowered = {c.lower(): c for c in train.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    m = re.search(r"target(?:\s+column)?[:\s]+['\"`]?([A-Za-z0-9_]+)", text, re.I)
    if m and m.group(1) in train.columns:
        return m.group(1)
    if only_in_train:
        return only_in_train[-1]
    return train.columns[-1]


def _guess_id(train: pd.DataFrame, test: pd.DataFrame) -> Optional[str]:
    for c in test.columns:
        if c not in train.columns:
            continue
        s = test[c]
        if s.is_unique and (
            "id" in c.lower() or s.dtype == object or np.issubdtype(s.dtype, np.integer)
        ) and train[c].is_unique:
            if "id" in c.lower() or c.lower() in ("index", "key", "row"):
                return c
    return None


def _infer_problem_type(y: pd.Series) -> tuple[str, int, Optional[list]]:
    y = y.dropna()
    nunique = y.nunique()
    is_float_like = (
        np.issubdtype(y.dtype, np.floating)
        and not np.all(np.isclose(y.dropna() % 1, 0))
    )
    if not is_float_like and nunique == 2:
        return "binary", 2, sorted(y.unique().tolist())
    if not is_float_like and (y.dtype == object or nunique <= max(20, int(0.05 * len(y)))):
        if y.dtype == object or nunique <= 50:
            return "multiclass", int(nunique), sorted(y.unique().tolist())
    return "regression", 1, None


def build_problem_spec(
    task_description: str,
    data_description: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> ProblemSpec:
    text = f"{task_description}\n{data_description}"
    target = _guess_target(train, test, text)
    if target not in train.columns:
        raise ValueError(f"Could not locate target column (guessed '{target}').")
    id_col = _guess_id(train, test)
    ptype, n_classes, labels = _infer_problem_type(train[target])
    return ProblemSpec(
        task_description=task_description.strip(),
        data_description=data_description.strip(),
        target_col=target,
        id_col=id_col,
        problem_type=ptype,
        n_classes=n_classes,
        class_labels=labels,
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
