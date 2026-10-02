"""
Synthetic reading generation for DEMO MODE presentations only.

Two things get generated here:
1. A flat, nameplate-matched "healthy baseline" seed window.
2. A scripted gradual fault-ramp sequence on top of that baseline.

The fault-ramp shapes are ported from data/raw/generate_sample.py's
scenario generators (same ramp curves, same relative magnitudes) so the
Bayesian fault-attribution priors in config/motor_spec.json still recognize
the pattern -- just compressed from a 300-row/50-day CSV into a handful of
steps for a live, few-seconds-per-step demo.

Nothing here writes to data/raw, models/, or config/motor_assets.json --
callers are responsible for keeping demo readings out of the real asset
store.
"""
import numpy as np

NOMINAL = {
    "vibration_mms": 0.9,
    "winding_temp_C": 78.0,
    "bearing_temp_C": 55.0,
    "power_factor": 0.87,
}

BASELINE_SEED_ROWS = 15

# A perfectly flat, all-identical baseline makes every rolling-window
# "suddenness ratio" feature a 0/0 divide (see feature_engineering.py's
# add_suddenness_features) -> NaN -> those rows get silently skipped by the
# scoring loop, so the baseline would show no health data at all instead of
# a green one. This jitter is just large enough to keep rolling std/slope
# non-degenerate, and small enough (+/-0.3%) that it still reads as "exactly
# at nameplate" and stays at the healthiest point in the trained IF/AE
# models' input distribution.
BASELINE_JITTER = 0.003


def _jitter(base, i, phase):
    return base * (1 + BASELINE_JITTER * np.sin(i * 0.9 + phase))


def baseline_row(rated_current, rated_voltage, rated_rpm, i):
    """One healthy-baseline reading, deterministic in `i` (no RNG state)."""
    return {
        "current_A": _jitter(rated_current, i, 0.0),
        "current_A_ph2": _jitter(rated_current, i, 2.1),
        "current_A_ph3": _jitter(rated_current, i, 4.2),
        "voltage_V": _jitter(rated_voltage, i, 1.0),
        "rpm": _jitter(rated_rpm, i, 3.0),
        "vibration_mms": _jitter(NOMINAL["vibration_mms"], i, 0.5),
        "winding_temp_C": _jitter(NOMINAL["winding_temp_C"], i, 1.5),
        "bearing_temp_C": _jitter(NOMINAL["bearing_temp_C"], i, 2.5),
        "power_factor": _jitter(NOMINAL["power_factor"], i, 0.2),
    }


def generate_baseline_seed(rated_current, rated_voltage, rated_rpm, n=BASELINE_SEED_ROWS):
    return [baseline_row(rated_current, rated_voltage, rated_rpm, i) for i in range(n)]


# Misalignment is deliberately excluded: its real signature is a sudden
# step-change (see generate_sample.py's gen_motor_misalignment), which
# conflicts with the "gradual, not an instant jump" requirement for this
# demo -- a step-change can't be shown "developing" over the ramp.
SCENARIOS = {
    "bearing_wear": {
        "label": "Bearing Wear",
        "signature": "Vibration and bearing temperature climb together while current stays flat.",
        "curve": 1.5,  # accelerating ramp, matches gen_motor_bearing_wear
    },
    "overload": {
        "label": "Overload",
        "signature": "Current and winding temperature climb together; RPM sags slightly.",
        "curve": 1.0,  # linear ramp, matches gen_motor_overload
    },
    "winding_fault_imbalance": {
        "label": "Winding Fault / Phase Imbalance",
        "signature": "One phase current drifts away from the other two; vibration stays flat.",
        "curve": 1.0,  # linear ramp, matches gen_motor_imbalance
    },
}


def generate_fault_sequence(rated_current, rated_voltage, rated_rpm, scenario, n_steps, seed):
    """
    Deterministic-per-seed ramp from onset (ramp=0) to full-fault (ramp=1)
    over n_steps, with realistic sensor noise on top so CUSUM/IF/AE have a
    real trend to react to instead of a suspiciously smooth line.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown demo fault_scenario '{scenario}'")

    rng = np.random.RandomState(seed)
    curve = SCENARIOS[scenario]["curve"]
    ramp = np.linspace(0, 1, n_steps) ** curve
    noise_scale = rated_current / 45.0  # same proportional scaling as generate_sample.py

    rows = []
    for i in range(n_steps):
        r = float(ramp[i])
        row = {
            "current_A": rated_current + rng.normal(0, 0.8 * noise_scale),
            "current_A_ph2": rated_current + rng.normal(0, 0.8 * noise_scale),
            "current_A_ph3": rated_current + rng.normal(0, 0.8 * noise_scale),
            "voltage_V": rated_voltage + rng.normal(0, 4),
            "rpm": rated_rpm + rng.normal(0, 3),
            "vibration_mms": max(0.1, NOMINAL["vibration_mms"] + rng.normal(0, 0.15)),
            "winding_temp_C": NOMINAL["winding_temp_C"] + rng.normal(0, 2),
            "bearing_temp_C": NOMINAL["bearing_temp_C"] + rng.normal(0, 2),
            "power_factor": NOMINAL["power_factor"] + rng.normal(0, 0.01),
        }
        if scenario == "bearing_wear":
            row["vibration_mms"] += r * 5.5
            row["bearing_temp_C"] += r * 30
        elif scenario == "overload":
            current_ramp = r * (rated_current * 12 / 45.0)
            row["current_A"] += current_ramp
            row["current_A_ph2"] += current_ramp
            row["current_A_ph3"] += current_ramp
            row["winding_temp_C"] += r * 40
            row["rpm"] -= r * 15
        elif scenario == "winding_fault_imbalance":
            row["current_A_ph3"] += r * (rated_current * 9 / 45.0)
        rows.append(row)
    return rows
