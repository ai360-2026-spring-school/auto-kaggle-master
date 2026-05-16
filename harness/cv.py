"""
The ground-truth evaluator.

This is the heart of the system and the direct analogue of the fixed 5-minute
training budget in autoresearch: it produces ONE comparable number per
experiment, computed identically every time, that the loop uses to keep or
discard the agent's work.

Leakage safety is enforced structurally (see contract.py): for every fold we
clone a fresh Solution, fit() it on the training rows only, then transform()
the held-out rows. The agent's code never sees a validation row at fit time.
The same fitted-per-fold solutions also produce out-of-fold predictions, which
are the only honest signal the agent gets to look at.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .contract import BaseSolution, validate_solution_output
from .metrics import Metric
from .model import LockedModel


@dataclass
class EvalResult:
    ok: bool
    score: float = float("nan")
    score_std: float = float("nan")
    fold_scores: list = field(default_factory=list)
    oof: Optional[np.ndarray] = None
    seconds: float = 0.0
    error: str = ""
    feature_importance: Optional[pd.Series] = None
    n_features: int = 0


def _make_folds(spec, n_splits, seed):
    y = spec.train[spec.target_col]
    if spec.problem_type in ("binary", "multiclass"):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(spec.train, y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(spec.train))


def _encode_target(y: pd.Series, spec) -> np.ndarray:
    if spec.problem_type == "regression":
        return y.to_numpy(dtype=float)
    mapping = {lab: i for i, lab in enumerate(spec.class_labels)}
    return y.map(mapping).to_numpy()


def evaluate_solution(
    solution: BaseSolution,
    spec,
    metric: Metric,
    cfg,
    time_limit: Optional[float] = None,
) -> EvalResult:
    """Run leakage-safe CV and return the single comparable score."""
    t0 = time.time()
    time_limit = time_limit or cfg.budget.max_seconds_per_eval
    folds = _make_folds(spec, cfg.budget.cv_folds, cfg.random_seed)

    y_all = spec.train[spec.target_col].reset_index(drop=True)
    y_enc = _encode_target(y_all, spec)
    n = len(spec.train)
    if spec.problem_type == "multiclass":
        oof = np.zeros((n, spec.n_classes))
    else:
        oof = np.zeros(n)

    fold_scores, importances = [], []
    feat_cols = None

    for k, (tr_idx, va_idx) in enumerate(folds):
        if time.time() - t0 > time_limit:
            return EvalResult(False, error=f"time limit exceeded after {k} folds",
                              seconds=time.time() - t0)
        tr_df = spec.train.iloc[tr_idx].reset_index(drop=True)
        va_df = spec.train.iloc[va_idx].reset_index(drop=True)
        y_tr = pd.Series(y_enc[tr_idx])
        y_va = y_enc[va_idx]

        sol = copy.deepcopy(solution)
        try:
            # fit() sees TRAIN ROWS ONLY — this is what makes CV honest.
            sol.fit(tr_df.drop(columns=[spec.target_col]),
                    pd.Series(y_all.iloc[tr_idx].values), spec)
            f_tr = sol.transform(tr_df.drop(columns=[spec.target_col]))
            f_va = sol.transform(va_df.drop(columns=[spec.target_col]))
        except Exception as e:  # noqa: BLE001 — surface to the agent
            return EvalResult(False, error=f"solution.fit/transform raised: {e!r}",
                              seconds=time.time() - t0)

        try:
            validate_solution_output(f_tr, f_va, spec)
        except Exception as e:  # noqa: BLE001
            return EvalResult(False, error=str(e), seconds=time.time() - t0)

        if feat_cols is None:
            feat_cols = list(f_tr.columns)

        model = LockedModel(spec.problem_type, spec.n_classes, cfg.catboost_params)
        try:
            model.fit(f_tr, y_tr.to_numpy(), f_va, y_va)
            raw = model.predict(f_va)
            pred = sol.postprocess(raw, va_df.drop(columns=[spec.target_col]))
        except Exception as e:  # noqa: BLE001
            return EvalResult(False, error=f"model/postprocess raised: {e!r}",
                              seconds=time.time() - t0)

        oof[va_idx] = pred
        importances.append(model.feature_importance())

        y_true_fold = y_all.iloc[va_idx].to_numpy()
        fold_scores.append(_score_fold(metric, y_true_fold, pred, spec))

    score = float(np.mean(fold_scores))
    fi = None
    if importances and importances[0] is not None:
        fi = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return EvalResult(
        ok=True,
        score=score,
        score_std=float(np.std(fold_scores)),
        fold_scores=[float(s) for s in fold_scores],
        oof=oof,
        seconds=time.time() - t0,
        feature_importance=fi,
        n_features=len(feat_cols or []),
    )


def _score_fold(metric: Metric, y_true, pred, spec) -> float:
    """Bridge raw predictions to the metric's expected input shape."""
    if metric.needs_proba:
        return metric.score(y_true, pred)
    if spec.problem_type == "binary" and not metric.needs_proba:
        # threshold probabilities for accuracy/f1-style metrics
        labels = np.array(spec.class_labels)
        return metric.score(y_true, labels[(np.asarray(pred) >= 0.5).astype(int)])
    if spec.problem_type == "multiclass" and not metric.needs_proba:
        labels = np.array(spec.class_labels)
        return metric.score(y_true, labels[np.argmax(pred, axis=1)])
    return metric.score(y_true, pred)


def is_improvement(candidate: EvalResult, incumbent: Optional[EvalResult],
                   metric: Metric, cfg) -> bool:
    """Accept only if it beats the incumbent by a noise-aware margin, so the
    loop chases real signal rather than CV variance."""
    if not candidate.ok:
        return False
    if incumbent is None or not incumbent.ok:
        return True
    sigma = max(incumbent.score_std, 1e-9)
    margin = cfg.budget.min_improvement_sigmas * sigma
    if metric.greater_is_better:
        return candidate.score > incumbent.score + margin
    return candidate.score < incumbent.score - margin
