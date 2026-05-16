"""
Metric registry. The competition metric is ground truth, exactly like
`val_bpb` in autoresearch. It is selected once from the task description and
then frozen; the agent optimizes against it but cannot redefine it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass
class Metric:
    name: str
    greater_is_better: bool
    needs_proba: bool
    fn: Callable  # (y_true, y_pred_or_proba) -> float

    def score(self, y_true, y_pred) -> float:
        return float(self.fn(y_true, y_pred))


def _rmse(yt, yp):
    return np.sqrt(skm.mean_squared_error(yt, yp))


def _rmsle(yt, yp):
    yp = np.clip(yp, 0, None)
    return np.sqrt(skm.mean_squared_error(np.log1p(yt), np.log1p(yp)))


def _binary_logloss(yt, yp):
    return skm.log_loss(yt, np.clip(yp, 1e-7, 1 - 1e-7), labels=[0, 1])


def _multi_logloss(yt, yp):
    return skm.log_loss(yt, yp)


_REGISTRY = {
    "rmse": Metric("rmse", False, False, _rmse),
    "mae": Metric("mae", False, False, skm.mean_absolute_error),
    "rmsle": Metric("rmsle", False, False, _rmsle),
    "r2": Metric("r2", True, False, skm.r2_score),
    "auc": Metric("auc", True, True, skm.roc_auc_score),
    "logloss": Metric("logloss", False, True, _binary_logloss),
    "accuracy": Metric("accuracy", True, False, skm.accuracy_score),
    "f1": Metric("f1", True, False,
                 lambda yt, yp: skm.f1_score(yt, yp, average="binary")),
    "f1_macro": Metric("f1_macro", True, False,
                        lambda yt, yp: skm.f1_score(yt, yp, average="macro")),
    "mlogloss": Metric("mlogloss", False, True, _multi_logloss),
}


def select_metric(task_description: str, problem_type: str) -> Metric:
    """Pick the metric named in the task text; otherwise a sane default."""
    t = task_description.lower()
    patterns = [
        (r"\brmsle\b|root mean squared log", "rmsle"),
        (r"\brmse\b|root mean squared error", "rmse"),
        (r"\bmae\b|mean absolute error", "mae"),
        (r"\br2\b|r-squared|coefficient of determination", "r2"),
        (r"\bauc\b|roc[ _-]?auc|area under", "auc"),
        (r"log[ _-]?loss|cross[ _-]?entropy|deviance", "logloss"),
        (r"macro[ _-]?f1|f1[ _-]?macro", "f1_macro"),
        (r"\bf1\b|f1[ _-]?score", "f1"),
        (r"\baccuracy\b|\bacc\b", "accuracy"),
    ]
    for pat, key in patterns:
        if re.search(pat, t):
            m = _REGISTRY[key]
            if key == "logloss" and problem_type == "multiclass":
                return _REGISTRY["mlogloss"]
            return m
    # Defaults when the description is silent.
    return {
        "binary": _REGISTRY["auc"],
        "multiclass": _REGISTRY["mlogloss"],
        "regression": _REGISTRY["rmse"],
    }[problem_type]
