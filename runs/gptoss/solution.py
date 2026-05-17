'''solution.py – extended feature engineering

This solution builds on the minimal baseline (which simply drops the id column
and passes through the raw features).  The baseline already achieves a solid
AUC (~0.918).  To push the score further we add a set of lightweight, leakage‑
free engineered features that are known to be useful for tabular data and that
the CatBoost model can exploit:

* **Frequency encoding** – count of each category value (Driver, Race,
  Compound).  This gives the model a numeric signal of how common a value is.
* **Smoothed target encoding** – a regularised mean of the target per category
  (global mean blended with the per‑category mean).  The smoothing factor is
  chosen to avoid over‑fitting on rare categories.
* **Log‑transform of skewed numerics** – columns with absolute skew > 0.5 are
  transformed with ``log1p`` to reduce long tails (PitStop, LapNumber, Stint,
  TyreLife, LapTime (s), RaceProgress, etc.).
* **Interaction features** – simple multiplicative interactions that capture
  domain knowledge (e.g. lap time multiplied by tyre life, lap number × tyre
  life, race progress × tyre life, lap time delta × tyre life, lap time ×
  lap number).  These are cheap to compute and often improve tree‑based
  models.

All statistics are learned *only* from the training data inside ``fit``;
``transform`` applies them deterministically to any dataframe (train folds or
test).  No target information leaks into the test set.

The postprocess step simply returns the raw predictions (CatBoost already
outputs probabilities), but we also clip them to the valid range [0, 1] as a
defensive measure.
''' 

from __future__ import annotations

import numpy as np
import pandas as pd

from harness import BaseSolution


class Solution(BaseSolution):
    def fit(self, train_df: pd.DataFrame, y: pd.Series, spec) -> None:
        # ------------------------------------------------------------------
        # 1. Basic column handling – drop id, keep everything else as raw features
        # ------------------------------------------------------------------
        self.drop_cols = []
        if spec.id_col and spec.id_col in train_df.columns:
            self.drop_cols.append(spec.id_col)
        # feature columns are everything except the id (target column is not in train_df)
        self.feature_cols = [c for c in train_df.columns if c not in self.drop_cols]

        # ------------------------------------------------------------------
        # 2. Identify column types
        # ------------------------------------------------------------------
        self.cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = train_df.select_dtypes(include=["number"]).columns.tolist()

        # ------------------------------------------------------------------
        # 3. Frequency encoding for categoricals
        # ------------------------------------------------------------------
        self.freq_maps: dict[str, dict] = {}
        for col in self.cat_cols:
            self.freq_maps[col] = train_df[col].value_counts().to_dict()

        # ------------------------------------------------------------------
        # 4. Smoothed target encoding (regularised mean per category)
        # ------------------------------------------------------------------
        self.global_mean = y.mean()
        self.te_maps: dict[str, dict] = {}
        smoothing = 10.0  # strength of regularisation
        for col in self.cat_cols:
            stats = pd.DataFrame({"cat": train_df[col], "target": y})
            agg = stats.groupby("cat")["target"].agg(["mean", "count"]).reset_index()
            weight = agg["count"] / (agg["count"] + smoothing)
            smoothed = weight * agg["mean"] + (1 - weight) * self.global_mean
            self.te_maps[col] = dict(zip(agg["cat"], smoothed))

        # ------------------------------------------------------------------
        # 5. Log‑transform for highly skewed numeric columns
        # ------------------------------------------------------------------
        skew = train_df[self.num_cols].skew().abs()
        self.log_cols = skew[skew > 0.5].index.tolist()

        # ------------------------------------------------------------------
        # 6. Interaction columns – defined once so transform knows which to create
        # ------------------------------------------------------------------
        self.interaction_defs: list[tuple[str, str, str]] = []  # (col_a, col_b, new_name)
        def add_inter(col_a: str, col_b: str, name: str | None = None):
            if col_a in self.feature_cols and col_b in self.feature_cols:
                new_name = name or f"{col_a}_x_{col_b}"
                self.interaction_defs.append((col_a, col_b, new_name))
        # Domain‑inspired interactions
        add_inter("LapTime (s)", "TyreLife")
        add_inter("LapNumber", "TyreLife")
        add_inter("RaceProgress", "TyreLife")
        add_inter("LapTime_Delta", "TyreLife")
        add_inter("LapTime (s)", "LapNumber")
        # Ratio interaction (safe division)
        if "TyreLife" in self.feature_cols:
            self.interaction_defs.append(("LapTime (s)", "TyreLife", "LapTime_per_TyreLife"))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Start with a copy of the raw feature columns (id already dropped)
        out = df[self.feature_cols].copy()

        # ------------------------------------------------------------------
        # Frequency encodings
        # ------------------------------------------------------------------
        for col in self.cat_cols:
            out[col + "_freq"] = out[col].map(self.freq_maps.get(col, {})).fillna(0)

        # ------------------------------------------------------------------
        # Smoothed target encodings
        # ------------------------------------------------------------------
        for col in self.cat_cols:
            out[col + "_te"] = out[col].map(self.te_maps.get(col, {})).fillna(self.global_mean)

        # ------------------------------------------------------------------
        # Log‑transformed numeric columns
        # ------------------------------------------------------------------
        for col in self.log_cols:
            out[col + "_log"] = np.log1p(out[col])

        # ------------------------------------------------------------------
        # Interaction features
        # ------------------------------------------------------------------
        for a, b, name in self.interaction_defs:
            if name == "LapTime_per_TyreLife":
                out[name] = out[a] / (out[b] + 1e-6)
            else:
                out[name] = out[a] * out[b]

        return out

    def postprocess(self, raw_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        # Clip predictions to [0, 1] for safety.
        return np.clip(raw_pred, 0.0, 1.0)
