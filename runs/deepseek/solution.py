"""
solution.py  —  THE ONLY FILE THE AGENT REWRITES.

This is the analogue of autoresearch's train.py. Everything about how data is
shaped before the locked CatBoost, and how predictions are shaped after, lives
here. The harness owns *when* these methods run (fold-safe), the agent owns
*what* they do.

This baseline is intentionally minimal: drop the id, pass features through,
identity postprocess. It establishes a valid, leak-free reference score that
every later experiment must beat. The agent replaces this file wholesale.
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
        
        # Learn early race threshold (30th percentile of RaceProgress)
        self.early_race_threshold = np.percentile(train_df['RaceProgress'].values, 30)
        
        # Store original column names for reference
        self.original_cols = [c for c in train_df.columns if c not in self.drop_cols]
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Start with original features (excluding id)
        features = df[[c for c in self.original_cols if c in df.columns]].copy()
        
        # Add engineered features
        # 1. Early race indicator (RaceProgress < 30th percentile from training)
        features['early_race'] = (df['RaceProgress'] < self.early_race_threshold).astype(int)
        
        # 2. Position gain indicator
        features['position_gain'] = (df['Position_Change'] > 0).astype(int)
        
        # 3. Early race position gain interaction
        features['early_race_position_gain'] = features['early_race'] * features['position_gain']
        
        # 4. Absolute position change (magnitude matters)
        features['position_change_abs'] = df['Position_Change'].abs()
        
        # 5. Zero-value indicators for key columns (threshold effects)
        features['LapTime_Delta_zero'] = (df['LapTime_Delta'] == 0).astype(int)
        features['Cumulative_Degradation_zero'] = (df['Cumulative_Degradation'] == 0).astype(int)
        features['Position_Change_zero'] = (df['Position_Change'] == 0).astype(int)
        features['PitStop_zero'] = (df['PitStop'] == 0).astype(int)
        
        return features

    def postprocess(self, raw_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        return raw_pred