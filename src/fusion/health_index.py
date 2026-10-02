"""
Combines every signal into one 0-100 health index + ranked fault causes.
This is the final output consumed by the API / frontend.
"""
import numpy as np

ZONE_PENALTY = {"normal": 0, "watch": 15, "alarm": 35, "critical": 60}


class HealthIndexFusion:
    def __init__(self, if_weight=0.25, ae_weight=0.25, cusum_weight=0.15, zone_weight=0.35):
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        self.cusum_weight = cusum_weight
        self.zone_weight = zone_weight

    def compute_zone_penalty(self, zone_classification: dict):
        """Worst-case zone across all parameters drives the base penalty."""
        zones = list(zone_classification.values())
        worst = max(zones, key=lambda z: ZONE_PENALTY[z])
        return ZONE_PENALTY[worst], worst

    def compute_health_index(self, zone_classification, if_score, ae_score, cusum_flagged_count, total_cusum_checks=5):
        zone_penalty, worst_zone = self.compute_zone_penalty(zone_classification)
        if_penalty = min(if_score, 1.0) * 100 * self.if_weight
        ae_penalty = min(ae_score, 2.0) / 2.0 * 100 * self.ae_weight
        cusum_penalty = (cusum_flagged_count / total_cusum_checks) * 100 * self.cusum_weight
        zone_penalty_weighted = zone_penalty * self.zone_weight * (100 / 60)  # rescale zone_penalty (max 60) to weighted 0-100 contribution

        total_penalty = if_penalty + ae_penalty + cusum_penalty + zone_penalty_weighted
        health_index = max(0, 100 - total_penalty)

        if health_index >= 85:
            severity = "Healthy"
        elif health_index >= 65:
            severity = "Watch"
        elif health_index >= 40:
            severity = "Investigate"
        else:
            severity = "Critical"

        return {
            "health_index": round(health_index, 1),
            "severity": severity,
            "worst_spec_zone": worst_zone,
            "contributing_scores": {
                "spec_zone_penalty": round(zone_penalty_weighted, 1),
                "isolation_forest_penalty": round(if_penalty, 1),
                "autoencoder_penalty": round(ae_penalty, 1),
                "cusum_drift_penalty": round(cusum_penalty, 1),
            },
        }

    def build_full_result(self, motor_id, timestamp, zone_classification, if_score,
                           ae_score, cusum_flagged_count, fault_ranking, reasoning_text):
        health = self.compute_health_index(zone_classification, if_score, ae_score, cusum_flagged_count)
        top_faults = [{"fault": f, "probability": round(p, 3)} for f, p in fault_ranking[:3]]
        return {
            "motor_id": motor_id,
            "timestamp": str(timestamp),
            **health,
            "zone_classification": zone_classification,
            "anomaly_scores": {
                "isolation_forest": round(float(if_score), 3),
                "autoencoder": round(float(ae_score), 3),
            },
            "likely_faults": top_faults,
            "reasoning": reasoning_text,
        }


def generate_reasoning_text(zone_classification, top_faults, observed_directions):
    trending_params = [p for p, d in observed_directions.items() if d != "stable"]
    if not top_faults or top_faults[0][1] < 0.15:
        return "No significant deviation pattern detected; readings within expected variation."

    top_fault, top_prob = top_faults[0]
    fault_readable = top_fault.replace("_", " ")
    if trending_params:
        params_text = ", ".join(trending_params)
        return (f"{fault_readable} suspected ({top_prob*100:.0f}% likelihood) — "
                f"driven by trending parameters: {params_text}.")
    return f"{fault_readable} suspected ({top_prob*100:.0f}% likelihood) based on current deviation pattern."
