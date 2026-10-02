"""
End-to-end scoring pipeline: raw reading(s) -> features -> spec zones,
Isolation Forest, Autoencoder, CUSUM, Bayesian fault attribution -> fused
health index + reasoning. This is what the API calls per motor.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np

from spec.spec_loader import MotorSpec
from features.feature_engineering import build_feature_set, FEATURE_COLUMNS
from detection.isolation_forest_model import AnomalyDetectorIF
from detection.autoencoder_model import AnomalyDetectorAE
from detection.cusum_drift import CusumDetector
from attribution.bayesian_network import FaultAttributionModel
from fusion.health_index import HealthIndexFusion, generate_reasoning_text

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "motor_spec.json"
ASSETS_PATH = Path(__file__).parent.parent.parent / "config" / "motor_assets.json"

DEFAULT_MOTOR_CLASS = "medium_rigid_15_75kw"
DEFAULT_RATED = {"rated_current_A": 45.0, "rated_voltage_V": 415.0, "rated_rpm": 1480}

SLOPE_COLS_MAP = {
    "vibration_mms": "vibration_deviation_slope",
    "bearing_temp_C": "bearing_temp_deviation_slope",
    "current_A": "current_deviation_slope",
    "winding_temp_C": "winding_temp_deviation_slope",
    "current_imbalance_pct": "imbalance_deviation_slope",
}

SUDDENNESS_COLS_MAP = {
    "vibration_mms": "vibration_deviation_suddenness_ratio",
    "bearing_temp_C": "bearing_temp_deviation_suddenness_ratio",
}

CUSUM_FEATURE_COLS = [
    "vibration_deviation", "current_deviation", "winding_temp_deviation",
    "bearing_temp_deviation", "imbalance_deviation",
]

LOCK_IN_FLOOR = 0.5        # probability floor granted to a locked-in fault
ONSET_CONFIRM_READINGS = 5  # elevated-zone readings averaged together before committing a lock
UNLOCK_NORMAL_STREAK = 5    # consecutive normal-zone readings required before releasing a lock


class MotorInferencePipeline:
    def __init__(self):
        with open(CONFIG_PATH) as f:
            self.fault_priors = json.load(f)["fault_priors"]
        if ASSETS_PATH.exists():
            with open(ASSETS_PATH) as f:
                self.asset_registry = json.load(f)
        else:
            self.asset_registry = {}
        self._specs = {}   # per motor_id: cached MotorSpec built from the asset registry

        self.if_model = AnomalyDetectorIF().load()
        self.ae_model = AnomalyDetectorAE(input_dim=len(FEATURE_COLUMNS)).load()
        self.cusum = CusumDetector(threshold=5.0, drift_allowance=0.5)
        self.fault_model = FaultAttributionModel(self.fault_priors)
        self.fusion = HealthIndexFusion()
        self._locked_diagnosis = {}   # per motor_id: {"fault": str} or None
        self._onset_buffer = {}       # per motor_id: list of fault_ranking dicts pending lock confirmation
        self._normal_streak = {}      # per motor_id: consecutive normal-zone readings while locked

    def get_spec(self, motor_id):
        """
        Builds (and caches) the MotorSpec for a given motor from its nameplate
        entry in config/motor_assets.json -- each motor gets its own rated
        current/voltage/RPM instead of one hardcoded value shared fleet-wide.
        Falls back to the original defaults for a motor with no registry entry
        (e.g. an ad-hoc /ingest call for a motor_id nobody configured yet).
        """
        if motor_id not in self._specs:
            asset = self.asset_registry.get(motor_id, {})
            self._specs[motor_id] = MotorSpec(
                motor_class=asset.get("motor_class", DEFAULT_MOTOR_CLASS),
                rated_current=asset.get("rated_current_A", DEFAULT_RATED["rated_current_A"]),
                rated_voltage=asset.get("rated_voltage_V", DEFAULT_RATED["rated_voltage_V"]),
                rated_rpm=asset.get("rated_rpm", DEFAULT_RATED["rated_rpm"]),
            )
        return self._specs[motor_id]

    def get_spec_table(self, motor_id):
        table = self.get_spec(motor_id).get_spec_table()
        asset = self.asset_registry.get(motor_id, {})
        if "power_kw" in asset:
            table["power_kw"] = asset["power_kw"]
        return table

    def score_motor_history(self, raw_df, motor_id, window=10):
        """
        raw_df: full raw telemetry for ONE motor, sorted by timestamp
        Returns a list of per-reading result dicts (health index, faults, reasoning)
        for every row once enough history exists (window size).
        """
        spec = self.get_spec(motor_id)
        feats = build_feature_set(raw_df, spec, window=window)
        feats = self.cusum.apply_to_df(feats, CUSUM_FEATURE_COLS)
        self._locked_diagnosis[motor_id] = None
        self._onset_buffer[motor_id] = []
        self._normal_streak[motor_id] = 0

        results = []
        for idx, row in feats.iterrows():
            if pd.isna(row[FEATURE_COLUMNS]).any():
                continue  # not enough history yet for this row's rolling features

            X_row = row[FEATURE_COLUMNS].values.reshape(1, -1).astype(float)
            if_score = self.if_model.score(X_row)[0]
            ae_score = self.ae_model.score(X_row)[0]

            zone_classification = spec.classify_reading(row)
            _, worst_zone = self.fusion.compute_zone_penalty(zone_classification)

            cusum_flag_cols = [f"{c}_cusum_flag" for c in CUSUM_FEATURE_COLS]
            cusum_flagged_count = sum(bool(row[c]) for c in cusum_flag_cols if c in row)

            observed_directions = self.fault_model.directions_from_row(row, SLOPE_COLS_MAP)
            suddenness = self.fault_model.suddenness_from_row(row, SUDDENNESS_COLS_MAP)
            fault_ranking = self.fault_model.predict_fault_probabilities(observed_directions, suddenness)
            fault_ranking = self._apply_diagnosis_lock(motor_id, fault_ranking, worst_zone)
            reasoning = generate_reasoning_text(zone_classification, fault_ranking, observed_directions)

            result = self.fusion.build_full_result(
                motor_id=motor_id,
                timestamp=row["timestamp"],
                zone_classification=zone_classification,
                if_score=if_score,
                ae_score=ae_score,
                cusum_flagged_count=cusum_flagged_count,
                fault_ranking=fault_ranking,
                reasoning_text=reasoning,
            )
            result["true_condition"] = row.get("true_condition", None)  # for validation only
            results.append(result)

        return results

    def _apply_diagnosis_lock(self, motor_id, fault_ranking, worst_zone):
        """
        Keeps a confirmed fault diagnosis from drifting once its onset
        step-change scrolls out of the rolling slope window. The earlier
        approach of counting N consecutive readings agreeing on a top fault
        was reverted -- adjacent rolling-window features are autocorrelated,
        so a noise streak can look identical to a real trend, and (separately
        tried) gating that on the CUSUM detector didn't hold up either: CUSUM
        in this codebase isn't currently a reliable "beyond the noise floor"
        signal (its fixed center=0.0 doesn't match several deviation
        features' true nominal value, so it drifts into false positives on
        long enough history regardless of any real fault) -- fixing that
        properly is a separate, fleet-wide-impact change, tracked separately.

        Instead, the trigger here is the spec-driven zone classification
        (ISO 10816-3 / NEMA MG-1 thresholds) -- an already-validated
        "significance test against the noise floor" that exists elsewhere in
        this pipeline. But a single elevated reading's top-ranked fault is
        still noisy on its own (the ranking flips between 2-4 candidates
        almost every reading even right at a real onset), so rather than
        locking onto whichever fault happens to be on top the instant the
        zone first leaves "normal", ONSET_CONFIRM_READINGS consecutive
        elevated readings are averaged together first -- that smooths out
        the per-reading flip-flopping while still drawing only on the onset
        window, not the drifted-later one. A brief dip back to "normal"
        resets an unconfirmed onset (spec noise, not a real recovery); once
        locked, only a sustained run of UNLOCK_NORMAL_STREAK normal readings
        releases it, so a single flickering "watch" reading near a zone
        boundary can't prematurely resurrect the same noise into a new lock.
        """
        locked = self._locked_diagnosis.get(motor_id)

        if locked:
            if worst_zone == "normal":
                self._normal_streak[motor_id] += 1
                if self._normal_streak[motor_id] >= UNLOCK_NORMAL_STREAK:
                    self._locked_diagnosis[motor_id] = None
                    locked = None
            else:
                self._normal_streak[motor_id] = 0

        if not locked:
            if worst_zone == "normal":
                self._onset_buffer[motor_id] = []
            else:
                buf = self._onset_buffer[motor_id]
                buf.append(dict(fault_ranking))
                if len(buf) >= ONSET_CONFIRM_READINGS:
                    avg_scores = {}
                    for scores in buf:
                        for fault, prob in scores.items():
                            avg_scores[fault] = avg_scores.get(fault, 0.0) + prob / len(buf)
                    winning_fault = max(avg_scores, key=avg_scores.get)
                    self._onset_buffer[motor_id] = []
                    if winning_fault != "normal_operation":
                        locked = {"fault": winning_fault}
                        self._locked_diagnosis[motor_id] = locked
                        self._normal_streak[motor_id] = 0

        if locked:
            return self.fault_model.apply_lock(fault_ranking, locked["fault"], LOCK_IN_FLOOR)

        return fault_ranking


if __name__ == "__main__":
    pipeline = MotorInferencePipeline()
    raw = pd.read_csv(
        Path(__file__).parent.parent.parent / "data" / "raw" / "sample_motor_data.csv",
        parse_dates=["timestamp"],
    )
    motor_df = raw[raw["motor_id"] == "MTR-002"].sort_values("timestamp").reset_index(drop=True)
    results = pipeline.score_motor_history(motor_df, "MTR-002")

    # print a few samples: before onset, right after onset, and near the end
    print(f"Total scored readings: {len(results)}")
    for i in [0, len(results)//2, len(results)-1]:
        r = results[i]
        print(f"\n--- reading {i} (true_condition={r['true_condition']}) ---")
        print(f"health_index={r['health_index']}  severity={r['severity']}  worst_zone={r['worst_spec_zone']}")
        print(f"top faults: {r['likely_faults']}")
        print(f"reasoning: {r['reasoning']}")
