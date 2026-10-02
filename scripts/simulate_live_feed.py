"""
Simulates a live sensor feed by POSTing readings to the /ingest endpoint
one at a time, with a delay between each — mimicking a real motor sending
telemetry every few seconds/minutes instead of a bulk CSV upload.

Usage:
    python simulate_live_feed.py --motor MTR-006 --scenario bearing_wear
    python simulate_live_feed.py --motor MTR-006 --scenario normal --delay 2
"""
import argparse
import time
import requests
import numpy as np
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

RATED_CURRENT = 45.0
RATED_VOLTAGE = 415.0
RATED_RPM = 1480


def make_reading(t, i, n_total, scenario):
    """Generate one reading at step i, following the same fault-injection
    logic as generate_sample.py, but yielded one row at a time."""
    base = {
        "timestamp": t.isoformat(),
        "current_A": RATED_CURRENT + np.random.normal(0, 0.8),
        "current_A_ph2": RATED_CURRENT + np.random.normal(0, 0.8),
        "current_A_ph3": RATED_CURRENT + np.random.normal(0, 0.8),
        "voltage_V": RATED_VOLTAGE + np.random.normal(0, 4),
        "rpm": RATED_RPM + np.random.normal(0, 3),
        "vibration_mms": max(0.1, 0.9 + np.random.normal(0, 0.15)),
        "winding_temp_C": 78 + np.random.normal(0, 2),
        "bearing_temp_C": 55 + np.random.normal(0, 2),
        "power_factor": 0.87 + np.random.normal(0, 0.01),
    }

    onset = n_total // 2
    if i < onset or scenario == "normal":
        return base

    progress = (i - onset) / max(1, (n_total - onset))

    if scenario == "bearing_wear":
        ramp = progress ** 1.5
        base["vibration_mms"] += ramp * 5.5
        base["bearing_temp_C"] += ramp * 30
    elif scenario == "overload":
        base["current_A"] += progress * 12
        base["current_A_ph2"] += progress * 12
        base["current_A_ph3"] += progress * 12
        base["winding_temp_C"] += progress * 40
        base["rpm"] -= progress * 15
    elif scenario == "misalignment":
        base["vibration_mms"] += 4.8  # step change, not ramp

    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor", default="MTR-006", help="Motor ID to simulate")
    parser.add_argument(
        "--scenario",
        choices=["normal", "bearing_wear", "overload", "misalignment"],
        default="bearing_wear",
    )
    parser.add_argument("--readings", type=int, default=40, help="Total readings to send")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between readings")
    args = parser.parse_args()

    t = datetime.now()
    print(f"Simulating live feed for {args.motor} — scenario: {args.scenario}")
    print(f"Sending {args.readings} readings, {args.delay}s apart. Ctrl+C to stop.\n")

    for i in range(args.readings):
        reading = make_reading(t, i, args.readings, args.scenario)
        try:
            resp = requests.post(f"{API_BASE}/assets/{args.motor}/ingest", json=reading, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if "health_index" in result:
                print(
                    f"[{i+1}/{args.readings}] health_index={result['health_index']} "
                    f"severity={result['severity']} "
                    f"top_fault={result.get('likely_faults', [{}])[0].get('fault', 'n/a')}"
                )
            else:
                print(f"[{i+1}/{args.readings}] {result.get('message', result)}")
        except requests.exceptions.RequestException as e:
            print(f"Error on reading {i+1}: {e}")
            break

        t += timedelta(hours=4)  # keep timestamps spaced like the sample data
        time.sleep(args.delay)

    print(f"\nDone. Check the frontend at http://localhost:5173/assets/{args.motor}")


if __name__ == "__main__":
    main()
