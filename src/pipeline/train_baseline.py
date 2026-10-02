"""
Trains the Isolation Forest and Autoencoder on healthy operating data.
Run this once initially, and periodically re-run as more confirmed-healthy
data accumulates.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from spec.spec_loader import MotorSpec
from features.feature_engineering import build_feature_set, FEATURE_COLUMNS
from detection.isolation_forest_model import AnomalyDetectorIF
from detection.autoencoder_model import AnomalyDetectorAE

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_motor_data.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def train_baseline_models(healthy_motor_ids=None, window=10):
    """
    healthy_motor_ids: list of motor_ids to use for training (should be
    confirmed-healthy periods only). If None, uses the first `window`
    readings of every motor (assumes early life = healthy, common
    real-world assumption before any fault has developed).
    """
    spec = MotorSpec()
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    all_feats = []
    for motor_id, group in raw.groupby("motor_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        feats = build_feature_set(group, spec, window=window)
        if healthy_motor_ids is not None:
            feats = feats[feats["motor_id"].isin(healthy_motor_ids)]
        else:
            # use only rows still labeled 'normal' in true_condition -- this
            # is our stand-in ground truth since we don't have real labels yet.
            # In production this would instead be "first N readings after
            # commissioning" or "AMC-confirmed healthy periods".
            feats = feats[feats["true_condition"] == "normal"]
        all_feats.append(feats)

    training_df = pd.concat(all_feats, ignore_index=True)
    training_df = training_df.dropna(subset=FEATURE_COLUMNS)
    X = training_df[FEATURE_COLUMNS].values

    print(f"Training on {len(X)} healthy readings across "
          f"{training_df['motor_id'].nunique()} motors")

    # --- Isolation Forest ---
    if_model = AnomalyDetectorIF(contamination=0.05)
    if_model.fit(X)
    if_model.save()
    print("Isolation Forest trained and saved.")

    # --- Autoencoder ---
    ae_model = AnomalyDetectorAE(input_dim=X.shape[1])
    ae_model.fit(X, epochs=150, batch_size=16, verbose=True)
    ae_model.save()
    print("Autoencoder trained and saved.")

    return if_model, ae_model


if __name__ == "__main__":
    train_baseline_models()
