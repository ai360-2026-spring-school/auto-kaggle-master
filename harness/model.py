"""
The frozen ML core.

This is the analogue of autoresearch's untouchable infrastructure. CatBoost is
chosen because it is a strong, robust default for tabular data with native
categorical handling and built-in early stopping. The config comes straight
from `config.LOCKED_CATBOST_PARAMS` and is identical in every experiment. The
agent cannot import or instantiate CatBoost itself — it only ever sees this
wrapper through the CV harness, so the model truly is a fixed black box.
"""
from __future__ import annotations

import functools
import os
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


@functools.lru_cache(maxsize=1)
def _detect_gpu() -> bool:
    """Probe whether CatBoost can use a GPU on this host.

    Result is cached for the process lifetime. The probe trains a tiny
    1-iteration model with task_type='GPU' and devices='0'; CatBoost raises
    immediately if no compatible CUDA device is available. Disabled by
    setting AUTOML_AGENT_CATBOOST_GPU=0.
    """
    if os.environ.get("AUTOML_AGENT_CATBOOST_GPU", "1") == "0":
        return False
    try:
        from catboost import CatBoostClassifier as _CC
        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 4))
        y = (X[:, 0] + rng.normal(scale=0.1, size=32) > 0).astype(int)
        _CC(iterations=1, task_type="GPU", devices="0",
            verbose=False, allow_writing_files=False).fit(X, y)
        return True
    except Exception:
        return False


class LockedModel:
    """Fixed CatBoost. fit() always uses an internal eval set for early stop.

    If a CUDA-capable GPU is present (auto-detected once per process), the
    model trains on GPU; otherwise it falls back to CPU. Behavior is
    overridable via AUTOML_AGENT_CATBOOST_GPU=0.
    """

    def __init__(self, problem_type: str, n_classes: int, params: dict):
        self.problem_type = problem_type
        self.n_classes = n_classes
        self.params = dict(params)
        self._model = None
        self._cat_features: list[str] = []
        if _detect_gpu():
            # CatBoost GPU mode rejects some CPU-only options. We strip them
            # defensively rather than baking GPU choices into config.py (which
            # must stay portable to CPU-only boxes).
            self.params["task_type"] = "GPU"
            self.params.setdefault("devices", "0")
            for k in ("subsample", "bootstrap_type", "random_strength",
                      "leaf_estimation_iterations"):
                self.params.pop(k, None)

    @staticmethod
    def _detect_cat_features(X: pd.DataFrame) -> list[str]:
        # Robust across pandas versions (incl. pandas>=3.0 StringDtype):
        # a column is categorical for CatBoost iff it is not numeric/datetime.
        cats = []
        for c in X.columns:
            s = X[c]
            if (pd.api.types.is_numeric_dtype(s)
                    and not pd.api.types.is_bool_dtype(s)):
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            cats.append(c)
        return cats

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self._cat_features:
            X[c] = X[c].astype(str).fillna("__nan__")
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray):
        self._cat_features = self._detect_cat_features(X)
        X, X_val = self._prep(X), self._prep(X_val)
        if self.problem_type == "regression":
            self._model = CatBoostRegressor(**self.params)
        else:
            self._model = CatBoostClassifier(
                **self.params,
                loss_function="Logloss" if self.problem_type == "binary"
                else "MultiClass",
            )
        train_pool = Pool(X, y, cat_features=self._cat_features)
        eval_pool = Pool(X_val, y_val, cat_features=self._cat_features)
        self._model.fit(train_pool, eval_set=eval_pool)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = self._prep(X)
        if self.problem_type == "regression":
            return self._model.predict(X)
        if self.problem_type == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict_proba(X)

    def feature_importance(self) -> Optional[pd.Series]:
        if self._model is None:
            return None
        return pd.Series(
            self._model.get_feature_importance(),
            index=self._model.feature_names_,
        ).sort_values(ascending=False)
