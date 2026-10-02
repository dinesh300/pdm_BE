"""
Tracks each motor's own rolling mean/std for deviation features — lets the
system flag "this motor is drifting toward the high end of its own normal
range" even when it's still technically inside the spec Normal zone.
This is intentionally lightweight (no full retraining), just an online
rolling statistic per asset.
"""
import pandas as pd


class AdaptiveBaseline:
    def __init__(self, window=20):
        self.window = window

    def compute(self, df, feature_cols):
        """
        Returns df with added *_baseline_mean, *_baseline_std, and
        *_baseline_zscore columns per feature — the zscore tells you how
        unusual the current reading is relative to THIS motor's own recent
        history, independent of the fixed spec zones.
        """
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                continue
            roll_mean = df[col].rolling(self.window, min_periods=5).mean()
            roll_std = df[col].rolling(self.window, min_periods=5).std().replace(0, 1e-6)
            df[f"{col}_baseline_mean"] = roll_mean
            df[f"{col}_baseline_zscore"] = (df[col] - roll_mean) / roll_std
        return df
