"""
CUSUM (Cumulative Sum) control chart — detects small SUSTAINED shifts in a
deviation feature, using the spec-defined nominal value as the center-line
(not a learned mean). This catches slow degradation creep before a value
crosses into Watch/Alarm zones.
"""
import numpy as np
import pandas as pd


class CusumDetector:
    def __init__(self, threshold=5.0, drift_allowance=0.5):
        """
        threshold: cumulative sum level that triggers a drift flag
        drift_allowance (k): slack parameter, filters out normal noise
        """
        self.threshold = threshold
        self.k = drift_allowance

    def compute(self, series, center=0.0):
        """
        series: pandas Series of a deviation feature (already spec-normalized,
                so center=0 means 'at nominal spec value')
        Returns: (cusum_pos, cusum_neg, flagged) arrays/series
        """
        s_pos = np.zeros(len(series))
        s_neg = np.zeros(len(series))
        values = series.fillna(center).values

        for i in range(1, len(values)):
            diff = values[i] - center
            s_pos[i] = max(0, s_pos[i-1] + diff - self.k)
            s_neg[i] = min(0, s_neg[i-1] + diff + self.k)

        flagged = (s_pos > self.threshold) | (s_neg < -self.threshold)
        return pd.Series(s_pos, index=series.index), pd.Series(s_neg, index=series.index), pd.Series(flagged, index=series.index)

    def apply_to_df(self, df, feature_cols):
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                continue
            s_pos, s_neg, flagged = self.compute(df[col], center=0.0)
            df[f"{col}_cusum_flag"] = flagged
        return df
