"""
Generates a sample motor telemetry CSV with embedded fault scenarios.
This is ONLY for testing the pipeline end-to-end since no real data exists yet.
Ground-truth 'true_condition' column is included for validation purposes only —
the detection engine must NOT read this column.

Each motor's rated current/voltage/RPM and injected scenario come from
config/motor_assets.json -- the nameplate registry is the single source of
truth for both the synthetic data here and the runtime MotorSpec instances.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ASSETS_PATH = Path(__file__).parent.parent.parent / "config" / "motor_assets.json"
OUT_PATH = Path(__file__).parent / "sample_motor_data.csv"

n_readings_per_motor = 300  # e.g., readings every 4 hours over ~50 days
timestamps = pd.date_range("2026-01-01", periods=n_readings_per_motor, freq="4h")


def gen_motor_normal(motor_id, rated_current, rated_voltage, rated_rpm):
    """Motor with no fault — stable operation with normal sensor noise."""
    noise_scale = rated_current / 45.0  # keep noise proportional to the motor's own scale
    df = pd.DataFrame({
        "timestamp": timestamps,
        "motor_id": motor_id,
        "current_A": rated_current + np.random.normal(0, 0.8 * noise_scale, n_readings_per_motor),
        "current_A_ph2": rated_current + np.random.normal(0, 0.8 * noise_scale, n_readings_per_motor),
        "current_A_ph3": rated_current + np.random.normal(0, 0.8 * noise_scale, n_readings_per_motor),
        "voltage_V": rated_voltage + np.random.normal(0, 4, n_readings_per_motor),
        "rpm": rated_rpm + np.random.normal(0, 3, n_readings_per_motor),
        "vibration_mms": np.clip(0.9 + np.random.normal(0, 0.15, n_readings_per_motor), 0.1, None),
        "winding_temp_C": 78 + np.random.normal(0, 2, n_readings_per_motor),
        "bearing_temp_C": 55 + np.random.normal(0, 2, n_readings_per_motor),
        "power_factor": 0.87 + np.random.normal(0, 0.01, n_readings_per_motor),
        "true_condition": "normal"
    })
    return df


def gen_motor_bearing_wear(motor_id, rated, onset_idx=150):
    """Bearing wear: vibration + bearing_temp gradually rise together after onset_idx, current stays flat."""
    df = gen_motor_normal(motor_id, *rated)
    n_after = n_readings_per_motor - onset_idx
    ramp = np.linspace(0, 1, n_after) ** 1.5  # accelerating degradation curve
    df.loc[onset_idx:, "vibration_mms"] += ramp * 5.5   # climbs into alarm/critical zone
    df.loc[onset_idx:, "bearing_temp_C"] += ramp * 30    # climbs toward limit
    df.loc[onset_idx:, "true_condition"] = "bearing_wear"
    df.loc[:onset_idx-1, "true_condition"] = "normal"
    return df


def gen_motor_overload(motor_id, rated, onset_idx=180):
    """Overload: current + winding_temp rise together, RPM sags slightly."""
    df = gen_motor_normal(motor_id, *rated)
    rated_current = rated[0]
    n_after = n_readings_per_motor - onset_idx
    ramp = np.linspace(0, 1, n_after)
    current_ramp = ramp * (rated_current * 12 / 45.0)  # scale overload magnitude to this motor's rating
    df.loc[onset_idx:, "current_A"] += current_ramp
    df.loc[onset_idx:, "current_A_ph2"] += current_ramp
    df.loc[onset_idx:, "current_A_ph3"] += current_ramp
    df.loc[onset_idx:, "winding_temp_C"] += ramp * 40
    df.loc[onset_idx:, "rpm"] -= ramp * 15
    df.loc[onset_idx:, "true_condition"] = "overload"
    df.loc[:onset_idx-1, "true_condition"] = "normal"
    return df


def gen_motor_imbalance(motor_id, rated, onset_idx=200):
    """Winding fault/imbalance: one phase current deviates from the other two, vibration stays flat."""
    df = gen_motor_normal(motor_id, *rated)
    rated_current = rated[0]
    n_after = n_readings_per_motor - onset_idx
    ramp = np.linspace(0, 1, n_after)
    df.loc[onset_idx:, "current_A_ph3"] += ramp * (rated_current * 9 / 45.0)  # phase 3 drifts up alone
    df.loc[onset_idx:, "true_condition"] = "winding_fault_imbalance"
    df.loc[:onset_idx-1, "true_condition"] = "normal"
    return df


def gen_motor_misalignment(motor_id, rated, onset_idx=220):
    """Misalignment: sudden (not gradual) vibration jump, RPM stable."""
    df = gen_motor_normal(motor_id, *rated)
    df.loc[onset_idx:, "vibration_mms"] += 4.8  # step change, not ramp
    df.loc[onset_idx:, "true_condition"] = "misalignment_looseness"
    df.loc[:onset_idx-1, "true_condition"] = "normal"
    return df


SCENARIO_GENERATORS = {
    "normal": lambda motor_id, rated: gen_motor_normal(motor_id, *rated),
    "bearing_wear": gen_motor_bearing_wear,
    "overload": gen_motor_overload,
    "winding_fault_imbalance": gen_motor_imbalance,
    "misalignment_looseness": gen_motor_misalignment,
}

with open(ASSETS_PATH) as f:
    assets = json.load(f)

all_dfs = []
for motor_id, asset in assets.items():
    rated = (asset["rated_current_A"], asset["rated_voltage_V"], asset["rated_rpm"])
    generator = SCENARIO_GENERATORS[asset["scenario"]]
    all_dfs.append(generator(motor_id, rated))

full_df = pd.concat(all_dfs, ignore_index=True)
full_df["current_imbalance_pct"] = (
    full_df[["current_A", "current_A_ph2", "current_A_ph3"]].max(axis=1) -
    full_df[["current_A", "current_A_ph2", "current_A_ph3"]].min(axis=1)
) / full_df[["current_A", "current_A_ph2", "current_A_ph3"]].mean(axis=1) * 100

cols = ["timestamp", "motor_id", "current_A", "current_A_ph2", "current_A_ph3",
        "current_imbalance_pct", "voltage_V", "rpm", "vibration_mms",
        "winding_temp_C", "bearing_temp_C", "power_factor", "true_condition"]
full_df = full_df[cols]
full_df.to_csv(OUT_PATH, index=False)
print(f"Generated {len(full_df)} rows across {full_df['motor_id'].nunique()} motors")
print(full_df.groupby("motor_id")["true_condition"].apply(lambda x: x.value_counts().to_dict()))
