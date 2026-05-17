"""
Pluggable agent backends.

`GigaChatAgent` / `YandexAgent` are the real paths: they talk to their
respective LLM APIs through the same backend-agnostic `ReActDriver`, so the
model can call EDA tools to gather evidence before emitting a complete new
solution.py.

`OfflineExpertAgent` is a deterministic stand-in that walks a curriculum of
strong, standard tabular-ML solutions. It needs no API key, makes the whole
pipeline runnable/testable offline, and serves as a competent baseline agent.

All backends satisfy the same `Agent.propose()` interface so run.py is
agnostic. Agents that don't use tools ignore the optional `context` kwarg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class Proposal:
    reasoning: str
    solution_source: str
    hypothesis: str = ""
    expected_effect: str = ""
    tool_trace: list[dict] = field(default_factory=list)


class Agent(Protocol):
    def propose(self, system_prompt: str, iteration_prompt: str,
                iteration: int, context: Optional[Any] = None) -> Proposal: ...


# --------------------------------------------------------------------------- #
#  Offline deterministic expert (curriculum of standard solutions)            #
# --------------------------------------------------------------------------- #
class OfflineExpertAgent:
    """Walks a fixed curriculum. Each step is a complete, leak-safe solution.py
    that layers one well-understood tabular technique on top of the last."""

    def __init__(self):
        self._curriculum = [
            ("Clean types + explicit missing indicators + median/mode impute. "
             "CatBoost handles raw categoricals, but explicit missingness flags "
             "and skew fixes usually help.", _SOL_CLEAN),
            ("Add frequency encoding for every categorical: count of each level "
             "learned on train folds only. Cheap, leak-safe, often strong.",
             _SOL_FREQ),
            ("Add out-of-fold-style target encoding computed in fit() from "
             "training rows only (the harness already isolates folds, so a "
             "smoothed target mean here is fold-safe).", _SOL_TARGET_ENC),
            ("Add numeric feature engineering: log1p on right-skewed positives, "
             "pairwise ratios/products of the top numeric features, row-level "
             "aggregates (mean/std/missing-count).", _SOL_NUMERIC),
            ("Add postprocessing: clip predictions to the observed train target "
             "range (regression) / keep proba calibrated; final consolidation.",
             _SOL_POSTPROC),
        ]

    def propose(self, system_prompt: str, iteration_prompt: str,
                iteration: int, context: Optional[Any] = None) -> Proposal:
        idx = min(iteration, len(self._curriculum) - 1)
        reasoning, source = self._curriculum[idx]
        return Proposal(reasoning=reasoning, solution_source=source.strip())


# ---- curriculum solution sources ----------------------------------------- #

_HEADER = '''\
from __future__ import annotations
import numpy as np
import pandas as pd
from harness import BaseSolution
'''

_SOL_CLEAN = _HEADER + '''

class Solution(BaseSolution):
    def fit(self, train_df, y, spec):
        self.drop_cols = [spec.id_col] if spec.id_col in train_df.columns else []
        df = train_df.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        self.medians = df[self.num_cols].median()
        self.skewed = [c for c in self.num_cols
                       if (df[c].dropna() > 0).all() and abs(df[c].skew()) > 1]

    def transform(self, df):
        df = df.drop(columns=self.drop_cols, errors="ignore").copy()
        for c in self.num_cols:
            df[c + "__isna"] = df[c].isna().astype(np.int8)
            df[c] = df[c].fillna(self.medians[c])
        for c in self.skewed:
            df[c + "__log1p"] = np.log1p(df[c].clip(lower=0))
        for c in self.cat_cols:
            df[c] = df[c].astype(str).fillna("__nan__")
        return df
'''

_SOL_FREQ = _HEADER + '''

class Solution(BaseSolution):
    def fit(self, train_df, y, spec):
        self.drop_cols = [spec.id_col] if spec.id_col in train_df.columns else []
        df = train_df.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        self.medians = df[self.num_cols].median()
        self.skewed = [c for c in self.num_cols
                       if (df[c].dropna() > 0).all() and abs(df[c].skew()) > 1]
        self.freq = {c: df[c].astype(str).value_counts(normalize=True)
                     for c in self.cat_cols}

    def transform(self, df):
        df = df.drop(columns=self.drop_cols, errors="ignore").copy()
        for c in self.num_cols:
            df[c + "__isna"] = df[c].isna().astype(np.int8)
            df[c] = df[c].fillna(self.medians[c])
        for c in self.skewed:
            df[c + "__log1p"] = np.log1p(df[c].clip(lower=0))
        for c in self.cat_cols:
            s = df[c].astype(str)
            df[c + "__freq"] = s.map(self.freq[c]).fillna(0.0)
            df[c] = s.fillna("__nan__")
        return df
'''

_SOL_TARGET_ENC = _HEADER + '''

class Solution(BaseSolution):
    def fit(self, train_df, y, spec):
        self.drop_cols = [spec.id_col] if spec.id_col in train_df.columns else []
        df = train_df.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        self.medians = df[self.num_cols].median()
        self.skewed = [c for c in self.num_cols
                       if (df[c].dropna() > 0).all() and abs(df[c].skew()) > 1]
        self.freq = {c: df[c].astype(str).value_counts(normalize=True)
                     for c in self.cat_cols}
        # smoothed target encoding (regression target or class index)
        yv = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        self.global_mean = float(yv.mean())
        self.te = {}
        for c in self.cat_cols:
            g = yv.groupby(df[c].astype(str).reset_index(drop=True))
            stats = g.agg(["mean", "count"])
            w = stats["count"] / (stats["count"] + 20.0)
            self.te[c] = (w * stats["mean"] + (1 - w) * self.global_mean)

    def transform(self, df):
        df = df.drop(columns=self.drop_cols, errors="ignore").copy()
        for c in self.num_cols:
            df[c + "__isna"] = df[c].isna().astype(np.int8)
            df[c] = df[c].fillna(self.medians[c])
        for c in self.skewed:
            df[c + "__log1p"] = np.log1p(df[c].clip(lower=0))
        for c in self.cat_cols:
            s = df[c].astype(str)
            df[c + "__freq"] = s.map(self.freq[c]).fillna(0.0)
            df[c + "__te"] = s.map(self.te[c]).fillna(self.global_mean)
            df[c] = s.fillna("__nan__")
        return df
'''

_SOL_NUMERIC = _HEADER + '''

class Solution(BaseSolution):
    def fit(self, train_df, y, spec):
        self.drop_cols = [spec.id_col] if spec.id_col in train_df.columns else []
        df = train_df.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        self.medians = df[self.num_cols].median()
        self.skewed = [c for c in self.num_cols
                       if (df[c].dropna() > 0).all() and abs(df[c].skew()) > 1]
        self.freq = {c: df[c].astype(str).value_counts(normalize=True)
                     for c in self.cat_cols}
        yv = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        self.global_mean = float(yv.mean())
        self.te = {}
        for c in self.cat_cols:
            g = yv.groupby(df[c].astype(str).reset_index(drop=True))
            stats = g.agg(["mean", "count"])
            w = stats["count"] / (stats["count"] + 20.0)
            self.te[c] = w * stats["mean"] + (1 - w) * self.global_mean
        # pick top numeric cols by variance for interactions
        var = df[self.num_cols].var().sort_values(ascending=False)
        self.top_num = var.head(5).index.tolist()

    def transform(self, df):
        df = df.drop(columns=self.drop_cols, errors="ignore").copy()
        for c in self.num_cols:
            df[c + "__isna"] = df[c].isna().astype(np.int8)
            df[c] = df[c].fillna(self.medians[c])
        for c in self.skewed:
            df[c + "__log1p"] = np.log1p(df[c].clip(lower=0))
        tn = self.top_num
        for i in range(len(tn)):
            for j in range(i + 1, len(tn)):
                a, b = tn[i], tn[j]
                df[f"{a}_x_{b}"] = df[a] * df[b]
                df[f"{a}_div_{b}"] = df[a] / (df[b].replace(0, np.nan))
        df["__row_isna"] = df[self.num_cols].isna().sum(axis=1) \
            if False else 0  # isna already imputed; keep placeholder stable
        df["__row_mean"] = df[tn].mean(axis=1)
        df["__row_std"] = df[tn].std(axis=1)
        for c in self.cat_cols:
            s = df[c].astype(str)
            df[c + "__freq"] = s.map(self.freq[c]).fillna(0.0)
            df[c + "__te"] = s.map(self.te[c]).fillna(self.global_mean)
            df[c] = s.fillna("__nan__")
        return df.replace([np.inf, -np.inf], 0).fillna(0) \
            if False else df.replace([np.inf, -np.inf], np.nan)
'''

_SOL_POSTPROC = _HEADER + '''

class Solution(BaseSolution):
    def fit(self, train_df, y, spec):
        self.spec_type = spec.problem_type
        self.drop_cols = [spec.id_col] if spec.id_col in train_df.columns else []
        df = train_df.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = [c for c in df.columns if c not in self.num_cols]
        self.medians = df[self.num_cols].median()
        self.skewed = [c for c in self.num_cols
                       if (df[c].dropna() > 0).all() and abs(df[c].skew()) > 1]
        self.freq = {c: df[c].astype(str).value_counts(normalize=True)
                     for c in self.cat_cols}
        yv = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        self.global_mean = float(yv.mean())
        self.y_lo, self.y_hi = float(yv.min()), float(yv.max())
        self.te = {}
        for c in self.cat_cols:
            g = yv.groupby(df[c].astype(str).reset_index(drop=True))
            stats = g.agg(["mean", "count"])
            w = stats["count"] / (stats["count"] + 20.0)
            self.te[c] = w * stats["mean"] + (1 - w) * self.global_mean
        var = df[self.num_cols].var().sort_values(ascending=False)
        self.top_num = var.head(5).index.tolist()

    def transform(self, df):
        df = df.drop(columns=self.drop_cols, errors="ignore").copy()
        for c in self.num_cols:
            df[c + "__isna"] = df[c].isna().astype(np.int8)
            df[c] = df[c].fillna(self.medians[c])
        for c in self.skewed:
            df[c + "__log1p"] = np.log1p(df[c].clip(lower=0))
        tn = self.top_num
        for i in range(len(tn)):
            for j in range(i + 1, len(tn)):
                a, b = tn[i], tn[j]
                df[f"{a}_x_{b}"] = df[a] * df[b]
                df[f"{a}_div_{b}"] = df[a] / (df[b].replace(0, np.nan))
        df["__row_mean"] = df[tn].mean(axis=1)
        df["__row_std"] = df[tn].std(axis=1)
        for c in self.cat_cols:
            s = df[c].astype(str)
            df[c + "__freq"] = s.map(self.freq[c]).fillna(0.0)
            df[c + "__te"] = s.map(self.te[c]).fillna(self.global_mean)
            df[c] = s.fillna("__nan__")
        return df.replace([np.inf, -np.inf], np.nan)

    def postprocess(self, raw_pred, df):
        if self.spec_type == "regression":
            pad = 0.02 * (self.y_hi - self.y_lo)
            return np.clip(raw_pred, self.y_lo - pad, self.y_hi + pad)
        return raw_pred
'''
