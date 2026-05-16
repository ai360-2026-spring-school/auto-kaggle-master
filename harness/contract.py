"""
The contract between the agent and the harness — the single most important
design decision in this system.

The user wants the agent to be able to "do anything to the data before the
model and anything to the predictions after". The danger of literally allowing
that is target leakage and train/test contamination, which would make the CV
score (our ground truth) a lie and the whole autonomous loop worthless.

We get full freedom *and* trustworthy CV by forcing the agent to express its
data work as a stateful estimator rather than a one-shot script:

    class Solution:
        def fit(self, train_df, y, spec) -> None
        def transform(self, df) -> pd.DataFrame      # features only, no target
        def postprocess(self, raw_pred, df) -> np.ndarray

The harness — not the agent — decides *when* fit/transform run. Inside CV it
calls fit() on the training folds only, then transform() on the held-out fold
and the test set. Any statistic the agent computes (target encodings, scalers,
imputation values, frequency maps, ...) is therefore learned without ever
seeing held-out rows. The agent can write arbitrarily creative feature code; it
physically cannot leak, because it never gets to touch the validation rows
during fit.

`postprocess` is the symmetric freedom on the output side (clipping,
calibration, rank transforms, expm1, label remapping...). It is also fit-safe:
it only sees raw model predictions plus features, never the true target.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseSolution(ABC):
    """Agent-authored solutions subclass this. Default postprocess = identity."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, y: pd.Series, spec) -> None:
        """Learn every data-dependent statistic here, from train rows only."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a feature frame. Must NOT contain the target or id columns.

        Must be pure given the state learned in fit(): calling transform on the
        same rows twice yields the same output, and it never reads `y`.
        """

    def postprocess(self, raw_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Optionally transform model output (calibration, clipping, ...)."""
        return raw_pred


def validate_solution_output(
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    spec,
) -> None:
    """Cheap structural guards run before any CatBoost fit. These catch the
    obvious leak/format mistakes early and give the agent a clear error to
    react to, instead of a misleadingly good score."""
    if spec.target_col in feat_train.columns:
        raise ValueError(
            f"LEAK: target '{spec.target_col}' present in transformed features."
        )
    if spec.id_col and spec.id_col in feat_train.columns:
        raise ValueError(
            f"id column '{spec.id_col}' must be dropped from features."
        )
    if list(feat_train.columns) != list(feat_test.columns):
        raise ValueError(
            "train/test feature columns differ:\n"
            f"  only in train: {set(feat_train.columns) - set(feat_test.columns)}\n"
            f"  only in test:  {set(feat_test.columns) - set(feat_train.columns)}"
        )
    if feat_train.shape[1] == 0:
        raise ValueError("transform() produced zero feature columns.")
    bad = [c for c in feat_train.columns
           if feat_train[c].isna().all()]
    if bad:
        raise ValueError(f"all-NaN feature columns: {bad}")
