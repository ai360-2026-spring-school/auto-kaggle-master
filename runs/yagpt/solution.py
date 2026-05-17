"""
solution.py — THE ONLY FILE THE AGENT REWRITES.

This is the analogue of autoresearch's train.py. Everything about how data is
shaped before the locked CatBoost, and how predictions are shaped after, lives
here. The harness owns *when* these methods run (fold-safe), the agent owns
*what* they do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import BaseSolution


class Solution(BaseSolution):
    def fit(self, train_df: pd.DataFrame, y: pd.Series, spec) -> None:
        # Remember which columns to drop so transform() is stateless wrt rows.
        self.drop_cols = []
        if spec.id_col and spec.id_col in train_df.columns:
            self.drop_cols.append(spec.id_col)
        
        # Learn frequency encoding for Driver
        self.driver_freq = train_df['Driver'].value_counts(normalize=True).to_dict()
        
        self.feature_cols = [c for c in train_df.columns if c not in self.drop_cols]
        self.feature_cols.append('DriverFreq')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self.feature_cols].copy()
        # Apply frequency encoding
        df['DriverFreq'] = df['Driver'].map(self.driver_freq).fillna(0)
        return df

    def postprocess(self, raw_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        return raw_pred