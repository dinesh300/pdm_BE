"""
Loads the motor spec config and classifies raw sensor readings into
Normal/Watch/Alarm/Critical zones based on standard-derived thresholds
(ISO 10816-3, NEMA MG-1). This is the deterministic "physical safety floor"
layer — always runs regardless of what the ML layers say.
"""
import json
from pathlib import Path

SPEC_PATH = Path(__file__).parent.parent.parent / "config" / "motor_spec.json"


class MotorSpec:
    def __init__(self, motor_class="medium_rigid_15_75kw", rated_current=45.0,
                 rated_voltage=415.0, rated_rpm=1480, spec_path=SPEC_PATH):
        with open(spec_path) as f:
            full_spec = json.load(f)
        self.class_spec = full_spec["motor_classes"][motor_class]
        self.fault_priors = full_spec["fault_priors"]
        self.rated_current = rated_current
        self.rated_voltage = rated_voltage
        self.rated_rpm = rated_rpm

    def _zone_from_bands(self, value, bands):
        """Given value and a dict of {zone_name: {min, max}}, return matching zone name."""
        for zone in ["normal", "watch", "alarm", "critical"]:
            band = bands[zone]
            lo = band["min"] if band["min"] is not None else float("-inf")
            hi = band["max"] if band["max"] is not None else float("inf")
            if lo <= value < hi or (zone == "critical" and value >= lo):
                return zone
        return "critical"  # fallback if value exceeds all bands

    def classify_vibration(self, value_mms):
        bands = self.class_spec["parameters"]["vibration_mms"]["zones"]
        return self._zone_from_bands(value_mms, bands)

    def classify_current(self, value_A):
        pct_of_rated = value_A / self.rated_current
        bands = self.class_spec["parameters"]["current_A"]["zones_pct_of_rated"]
        return self._zone_from_bands(pct_of_rated, bands)

    def classify_voltage(self, value_V):
        pct_of_rated = value_V / self.rated_voltage
        bands = self.class_spec["parameters"]["voltage_V"]["zones_pct_of_rated"]
        return self._zone_from_bands(pct_of_rated, bands)

    def classify_winding_temp(self, value_C):
        limit = self.class_spec["parameters"]["winding_temp_C"]["limit_c"]
        pct_of_limit = value_C / limit
        bands = self.class_spec["parameters"]["winding_temp_C"]["zones_pct_of_limit"]
        return self._zone_from_bands(pct_of_limit, bands)

    def classify_bearing_temp(self, value_C):
        limit = self.class_spec["parameters"]["bearing_temp_C"]["limit_c"]
        pct_of_limit = value_C / limit
        bands = self.class_spec["parameters"]["bearing_temp_C"]["zones_pct_of_limit"]
        return self._zone_from_bands(pct_of_limit, bands)

    def classify_current_imbalance(self, value_pct):
        bands = self.class_spec["parameters"]["current_imbalance_pct"]["zones"]
        return self._zone_from_bands(value_pct, bands)

    def classify_reading(self, row):
        """Classify a full reading (dict or pandas Series) across all parameters."""
        return {
            "vibration_mms": str(self.classify_vibration(row["vibration_mms"])),
            "current_A": str(self.classify_current(row["current_A"])),
            "voltage_V": str(self.classify_voltage(row["voltage_V"])),
            "winding_temp_C": str(self.classify_winding_temp(row["winding_temp_C"])),
            "bearing_temp_C": str(self.classify_bearing_temp(row["bearing_temp_C"])),
            "current_imbalance_pct": str(self.classify_current_imbalance(row["current_imbalance_pct"])),
        }

    def get_spec_table(self):
        """Returns the full threshold table for frontend display purposes."""
        return {
            "motor_class": self.class_spec["description"],
            "insulation_class": self.class_spec["insulation_class"],
            "rated_current_A": self.rated_current,
            "rated_voltage_V": self.rated_voltage,
            "rated_rpm": self.rated_rpm,
            "parameters": self.class_spec["parameters"],
        }


if __name__ == "__main__":
    # quick smoke test
    spec = MotorSpec()
    test_row = {
        "vibration_mms": 5.2, "current_A": 46, "voltage_V": 415,
        "winding_temp_C": 82, "bearing_temp_C": 88, "current_imbalance_pct": 0.5
    }
    print(spec.classify_reading(test_row))
