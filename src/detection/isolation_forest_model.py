"""
Isolation Forest trained on spec-normalized deviation features.
Catches unusual COMBINATIONS of deviations that per-parameter zone checks
would miss (e.g., several parameters mildly elevated together = statistically
unusual even if each one alone is still 'Normal').
"""
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "isolation_forest.pkl"


class AnomalyDetectorIF:
    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X):
        """X: dataframe or array of feature columns, ideally mostly-healthy data."""
        X = np.nan_to_num(X, nan=0.0)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def score(self, X):
        """
        Returns anomaly scores normalized to 0-1 (higher = more anomalous).
        sklearn's decision_function: higher = more normal, so we invert & rescale.
        """
        X = np.nan_to_num(X, nan=0.0)
        raw_scores = self.model.decision_function(X)  # higher = more normal
        # rescale: typical range is roughly [-0.5, 0.5] -> map to [1, 0]
        normalized = np.clip(0.5 - raw_scores, 0, 1)
        return normalized

    def save(self, path=MODEL_PATH):
        joblib.dump(self.model, path)

    def load(self, path=MODEL_PATH):
        self.model = joblib.load(path)
        self.is_fitted = True
        return self
