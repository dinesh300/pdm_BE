"""
Probabilistic fault attribution. Given an anomalous reading's deviation
pattern (which parameters are trending, in what direction, together or
independently), computes P(fault_type | observed_pattern).

v1 approach: a discrete evidence-scoring model seeded with domain-expert
priors from motor_spec.json's fault_priors block. This is deliberately
simpler than a full pgmpy CPD-based network for the first pass (small
labeled dataset), but structured so it can be upgraded to a learned
pgmpy BayesianNetwork once real confirmed-fault outcomes accumulate via
the AMC ground-truth loop -- the interface (predict_fault_probabilities)
stays the same either way.
"""
import numpy as np


class FaultAttributionModel:
    def __init__(self, fault_priors):
        """fault_priors: the fault_priors dict from motor_spec.json"""
        self.fault_priors = fault_priors

    def _direction(self, slope, threshold=0.005):
        if slope is None or np.isnan(slope):
            return "stable"
        if slope > threshold:
            return "increasing"
        if slope < -threshold:
            return "decreasing"
        return "stable"

    def _matches_signature(self, observed, signature, suddenness=None):
        """
        observed: dict of {parameter: direction} derived from current row's slopes
        signature: the fault's expected pattern from fault_priors
        suddenness: optional dict of {parameter: suddenness_ratio (0-1)} — high
                    ratio means the recent change happened in a single step
                    rather than spread gradually across the window.
        Returns a match score 0-1 (fraction of signature conditions satisfied).
        """
        if not signature:
            return 0.3  # baseline "normal_operation" catch-all score
        matches = 0
        total = len(signature)
        suddenness = suddenness or {}
        for param, expected_dir in signature.items():
            obs_dir = observed.get(param, "stable")
            if expected_dir == "stable_or_decreasing" and obs_dir in ("stable", "decreasing"):
                matches += 1
            elif expected_dir == "sudden_increase":
                is_sudden = suddenness.get(param, 0.0) > 0.55
                if obs_dir == "increasing" and is_sudden:
                    matches += 1
                elif obs_dir == "increasing" and not is_sudden:
                    matches += 0.3  # direction right, onset shape wrong -- partial credit only
            elif expected_dir == "increasing" and obs_dir == "increasing":
                # for gradual-onset faults, penalize if the change was actually sudden
                is_sudden = suddenness.get(param, 0.0) > 0.55
                matches += 0.5 if is_sudden else 1.0
            elif obs_dir == expected_dir:
                matches += 1
        return matches / total

    def predict_fault_probabilities(self, observed_directions, suddenness=None):
        """
        observed_directions: dict like {
            'vibration_mms': 'increasing', 'bearing_temp_C': 'increasing',
            'current_A': 'stable', ...
        }
        suddenness: optional dict {'vibration_mms': 0.0-1.0, ...}
        Returns ranked list of {fault_name: probability}, normalized to sum to 1.
        """
        scores = {}
        for fault_name, fault_info in self.fault_priors.items():
            match_score = self._matches_signature(observed_directions, fault_info["signature"], suddenness)
            prior = fault_info["prior_probability"]
            # Bayesian-style: posterior proportional to likelihood * prior
            scores[fault_name] = match_score * prior

        total = sum(scores.values()) or 1e-9
        probabilities = {k: v / total for k, v in scores.items()}
        ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def directions_from_row(self, row, slope_cols_map):
        """
        Helper: build the observed_directions dict from a feature row's slope columns.
        slope_cols_map: {'vibration_mms': 'vibration_deviation_slope', ...}
        """
        return {
            param: self._direction(row.get(slope_col))
            for param, slope_col in slope_cols_map.items()
        }

    def suddenness_from_row(self, row, suddenness_cols_map):
        """
        Helper: build the suddenness dict from a feature row's suddenness_ratio columns.
        suddenness_cols_map: {'vibration_mms': 'vibration_deviation_suddenness_ratio', ...}
        """
        return {
            param: row.get(col, 0.0)
            for param, col in suddenness_cols_map.items()
        }

    def apply_lock(self, fault_ranking, locked_fault, min_probability=0.5):
        """
        Raises a previously-confirmed diagnosis back to the top of the ranking
        when CUSUM evidence (see live_feed_inference.py) shows the underlying
        shift is still statistically present, even though the rolling-window
        slope that originally surfaced it has scrolled out of the feature
        window. Only raises the floor for that fault -- never fabricates
        evidence beyond what current CUSUM state supports, and never touches
        the ranking if the fault is already at/above the floor on its own.
        """
        scores = dict(fault_ranking)
        if locked_fault not in scores or scores[locked_fault] >= min_probability:
            return fault_ranking
        other_total = sum(v for k, v in scores.items() if k != locked_fault) or 1e-9
        remaining = 1 - min_probability
        boosted = {
            k: (min_probability if k == locked_fault else v / other_total * remaining)
            for k, v in scores.items()
        }
        return sorted(boosted.items(), key=lambda x: x[1], reverse=True)
