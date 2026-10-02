"""
Converts raw motor telemetry into spec-normalized deviation features.
This is the foundation of the "intelligence" layer — every parameter is put
onto a comparable 0-1+ scale relative to its spec limit, so the ML models
downstream learn deviation PATTERNS rather than raw sensor magnitudes.
"""
import pandas as pd
import numpy as np


def compute_deviation_ratio(value, rated_or_zero, limit):
    """
    deviation_ratio = (value - rated) / (limit - rated)
    0 = exactly at rated/nominal, 1 = exactly at the physical limit, >1 = beyond limit.
    """
    denom = (limit - rated_or_zero)
    if denom == 0:
        return 0.0
    return (value - rated_or_zero) / denom


def add_deviation_features(df, spec):
    """
    df: raw telemetry dataframe (single motor, sorted by timestamp)
    spec: MotorSpec instance
    Returns df with added *_deviation columns (spec-normalized, comparable scale).
    """
    df = df.copy()

    df["vibration_deviation"] = df["vibration_mms"].apply(
        lambda v: compute_deviation_ratio(v, 0, spec.class_spec["parameters"]["vibration_mms"]["zones"]["alarm"]["max"])
    )
    df["current_deviation"] = df["current_A"].apply(
        lambda v: compute_deviation_ratio(v, spec.rated_current, spec.rated_current * 1.20)
    )
    df["winding_temp_deviation"] = df["winding_temp_C"].apply(
        lambda v: compute_deviation_ratio(v, 0, spec.class_spec["parameters"]["winding_temp_C"]["limit_c"])
    )
    df["bearing_temp_deviation"] = df["bearing_temp_C"].apply(
        lambda v: compute_deviation_ratio(v, 0, spec.class_spec["parameters"]["bearing_temp_C"]["limit_c"])
    )
    df["imbalance_deviation"] = df["current_imbalance_pct"].apply(
        lambda v: compute_deviation_ratio(v, 0, spec.class_spec["parameters"]["current_imbalance_pct"]["zones"]["alarm"]["max"])
    )
    return df


def add_trend_features(df, window=10):
    """
    Adds rolling slope (rate of change) and rolling volatility (std dev) per
    deviation feature — this is what lets CUSUM/ML distinguish 'noisy but flat'
    from 'genuinely trending toward failure'.
    """
    df = df.copy()
    deviation_cols = [c for c in df.columns if c.endswith("_deviation")]

    for col in deviation_cols:
        # rolling slope via simple linear fit over the window
        df[f"{col}_slope"] = df[col].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=False
        )
        df[f"{col}_volatility"] = df[col].rolling(window).std()

    return df


def add_suddenness_features(df, window=10):
    """
    Distinguishes a SUDDEN step-change (e.g., misalignment) from a GRADUAL
    ramp (e.g., bearing wear) that would otherwise look identical to a plain
    slope check. Computes what fraction of the total window's movement
    happened in a single step vs spread evenly across the window.
    A ratio near 1 = sudden onset; near 1/window = gradual/linear ramp.
    """
    df = df.copy()
    deviation_cols = [c for c in df.columns if c.endswith("_deviation")]
    for col in deviation_cols:
        max_abs_step = df[col].diff().abs().rolling(window).max()
        window_range = (df[col].rolling(window).max() - df[col].rolling(window).min()).replace(0, np.nan)
        df[f"{col}_suddenness_ratio"] = (max_abs_step / window_range).clip(0, 1)
    return df


def add_cross_correlation_features(df, pairs, window=10):
    """
    Rolling correlation between parameter pairs — key for joint-pattern
    reasoning (e.g., does vibration rise TOGETHER with bearing temp,
    which is the bearing-wear signature, vs independently).
    """
    df = df.copy()
    for (a, b) in pairs:
        col_name = f"corr_{a}_{b}"
        df[col_name] = df[a].rolling(window).corr(df[b])
    return df


def build_feature_set(df, spec, window=10):
    """Full pipeline: raw df -> deviation features -> trend features -> cross-correlation features."""
    df = add_deviation_features(df, spec)
    df = add_trend_features(df, window=window)
    df = add_suddenness_features(df, window=window)
    df = add_cross_correlation_features(
        df,
        pairs=[
            ("vibration_deviation", "bearing_temp_deviation"),   # bearing wear signature
            ("current_deviation", "winding_temp_deviation"),     # overload signature
        ],
        window=window,
    )
    return df


FEATURE_COLUMNS = [
    "vibration_deviation", "current_deviation", "winding_temp_deviation",
    "bearing_temp_deviation", "imbalance_deviation",
    "vibration_deviation_slope", "current_deviation_slope",
    "winding_temp_deviation_slope", "bearing_temp_deviation_slope",
    "imbalance_deviation_slope",
    "vibration_deviation_volatility", "current_deviation_volatility",
    "winding_temp_deviation_volatility", "bearing_temp_deviation_volatility",
    "imbalance_deviation_volatility",
    "vibration_deviation_suddenness_ratio", "bearing_temp_deviation_suddenness_ratio",
    "corr_vibration_deviation_bearing_temp_deviation",
    "corr_current_deviation_winding_temp_deviation",
]


if __name__ == "__main__":
    import sys
    sys.path.append(str(__import__("pathlib").Path(__file__).parent.parent))
    from spec.spec_loader import MotorSpec

    spec = MotorSpec()
    raw = pd.read_csv("/home/claude/motor-pdm-intelligence/data/raw/sample_motor_data.csv")
    single_motor = raw[raw["motor_id"] == "MTR-002"].reset_index(drop=True)
    feats = build_feature_set(single_motor, spec)
    print(feats[["timestamp"] + FEATURE_COLUMNS].tail(10))
