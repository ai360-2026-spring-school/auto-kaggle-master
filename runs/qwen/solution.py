from __future__ import annotations

import numpy as np
import pandas as pd

from harness import BaseSolution


class Solution(BaseSolution):
    def __init__(self) -> None:
        super().__init__()
        self.target_enc_map_ = None
        self.global_target_mean_ = None
        self.drop_cols = []
        self.feature_cols = []

    def fit(self, train_df: pd.DataFrame, y: pd.Series, spec) -> None:
        # Drop ID
        self.drop_cols = []
        if spec.id_col and spec.id_col in train_df.columns:
            self.drop_cols.append(spec.id_col)

        # Prepare working copy
        df = train_df.copy()
        df[spec.target_col] = y

        # Global target mean for smoothing
        self.global_target_mean_ = y.mean()

        # Smoothed target encoding for Driver
        # Use smoothing to avoid overfitting on rare drivers
        col = 'Driver'
        target = spec.target_col
        smoothing = 5.0  # Controls weight of global mean

        stats = df.groupby(col)[target].agg(['count', 'mean'])
        smoothed_mean = (stats['count'] * stats['mean'] + smoothing * self.global_target_mean_) / (stats['count'] + smoothing)
        self.target_enc_map_ = smoothed_mean.to_dict()

        # Keep all original features + new encoded feature
        self.feature_cols = [c for c in df.columns if c not in self.drop_cols and c != target]
        # The transform will add the encoded column; we don't remove the raw Driver here

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply transformations
        out_df = df.copy()
        
        # Drop columns marked in fit
        out_df = out_df.drop(columns=[c for c in self.drop_cols if c in out_df.columns])

        # Apply target encoding on Driver
        enc_col = 'Driver_target_enc'
        out_df[enc_col] = out_df['Driver'].map(self.target_enc_map_).fillna(self.global_target_mean_)

        return out_df

    def postprocess(self, raw_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        # Clip predictions to valid probability range
        return np.clip(raw_pred, 1e-6, 1 - 1e-6)
