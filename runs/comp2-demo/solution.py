from __future__ import annotations

import numpy as np
import pandas as pd

from harness import BaseSolution


class Solution(BaseSolution):
    def fit(self, train_df: pd.DataFrame, y: pd.Series, spec) -> None:
        # Normalize TyreLife during fitting
        tyre_life_mean = train_df["TyreLife"].mean()
        tyre_life_std = train_df["TyreLife"].std()
        self.tyre_life_normalizer = lambda x: (x - tyre_life_mean) / tyre_life_std
    
        # Create interaction term between Stint and RaceProgress
        self.stint_raceprogress_interaction = True
    
        # Remember which columns to drop so transform() is stateless wrt rows.
        self.drop_cols = []
        if spec.id_col and spec.id_col in train_df.columns:
            self.drop_cols.append(spec.id_col)
        self.feature_cols = [c for c in train_df.columns
                             if c not in self.drop_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize TyreLife
        df["TyreLife"] = self.tyre_life_normalizer(df["TyreLife"])
    
        # Create interaction term between Stint and RaceProgress
        if self.stint_raceprogress_interaction:
            df["Stint_RaceProgress"] = df["Stint"] * df["RaceProgress"]
    
        # Drop the id column if present
        return df.drop(columns=self.drop_cols, errors="ignore")

    def postprocess(self, raw_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        return raw_pred