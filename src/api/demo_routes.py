"""
DEMO MODE endpoints for client presentations.

A session clones the nameplate of a real asset into an in-memory-only demo
registry entry, seeds a healthy baseline, and lets the frontend step through
a pre-scripted gradual fault ramp one reading at a time (POST .../step).

Fully isolated from the real pipeline: separate in-memory stores below,
demo_id values are never written into config/motor_assets.json or the real
_asset_store/_scored_cache in main.py, and nothing here retrains or
persists to models/ or data/raw.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import random
import uuid
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from demo.scenario_generator import (
    BASELINE_SEED_ROWS, SCENARIOS, generate_baseline_seed, generate_fault_sequence,
)

router = APIRouter(prefix="/demo", tags=["demo"])

STEP_INTERVAL_MS = 1500

_pipeline = None

# All demo state lives here -- separate from main.py's _asset_store /
# _scored_cache, so a demo session can never read or write real asset data.
_demo_asset_store = {}   # demo_id -> DataFrame
_demo_scored_cache = {}  # demo_id -> list of scored results
_demo_scripts = {}       # demo_id -> list of not-yet-applied fault-ramp rows
_demo_cursor = {}        # demo_id -> int, index into _demo_scripts
_demo_sessions = {}      # demo_id -> {base_motor_id, fault_scenario, duration_seconds, next_timestamp}


def attach(pipeline):
    global _pipeline
    _pipeline = pipeline


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _with_imbalance(row):
    row = dict(row)
    currents = [row["current_A"], row["current_A_ph2"], row["current_A_ph3"]]
    row["current_imbalance_pct"] = (max(currents) - min(currents)) / (sum(currents) / 3) * 100
    return row


class DemoStartRequest(BaseModel):
    base_motor_id: str
    fault_scenario: str = "bearing_wear"
    duration_seconds: int = 45


def _rated_nameplate(base_motor_id):
    asset = _pipeline.asset_registry.get(base_motor_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Unknown base_motor_id '{base_motor_id}'")
    return asset


def _init_session(demo_id, base_motor_id, fault_scenario, duration_seconds):
    if fault_scenario not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown fault_scenario '{fault_scenario}'. Choose one of: {list(SCENARIOS)}",
        )
    asset = _rated_nameplate(base_motor_id)
    rated_current = asset["rated_current_A"]
    rated_voltage = asset["rated_voltage_V"]
    rated_rpm = asset["rated_rpm"]

    # Clone the nameplate for spec/display purposes only -- in-memory, keyed
    # under the demo_id, never written back to the real registry entry.
    _pipeline.asset_registry[demo_id] = {**asset, "scenario": "demo"}

    n_steps = max(10, round(duration_seconds * 1000 / STEP_INTERVAL_MS))
    seed = random.randint(0, 2**31 - 1)
    baseline_rows = [_with_imbalance(r) for r in
                      generate_baseline_seed(rated_current, rated_voltage, rated_rpm, BASELINE_SEED_ROWS)]
    fault_rows = [_with_imbalance(r) for r in
                   generate_fault_sequence(rated_current, rated_voltage, rated_rpm, fault_scenario, n_steps, seed)]

    now = pd.Timestamp.now()
    step = pd.Timedelta(minutes=5)
    for i, row in enumerate(baseline_rows):
        row["timestamp"] = now - (BASELINE_SEED_ROWS - i) * step

    df = pd.DataFrame(baseline_rows)
    df["motor_id"] = demo_id
    df["true_condition"] = "normal"

    _demo_asset_store[demo_id] = df
    _demo_scripts[demo_id] = fault_rows
    _demo_cursor[demo_id] = 0
    _demo_sessions[demo_id] = {
        "base_motor_id": base_motor_id,
        "fault_scenario": fault_scenario,
        "duration_seconds": duration_seconds,
        "next_timestamp": now,
    }

    scored = _pipeline.score_motor_history(df, demo_id)
    _demo_scored_cache[demo_id] = scored

    return {
        "demo_id": demo_id,
        "base_motor_id": base_motor_id,
        "fault_scenario": fault_scenario,
        "scenario_label": SCENARIOS[fault_scenario]["label"],
        "total_steps": n_steps,
        "step_interval_ms": STEP_INTERVAL_MS,
        "step": 0,
        "done": False,
        "spec": _sanitize(_pipeline.get_spec_table(demo_id)),
        "health": _sanitize(scored[-1]) if scored else None,
    }


@router.get("/scenarios")
def list_scenarios():
    return {key: {"label": v["label"], "signature": v["signature"]} for key, v in SCENARIOS.items()}


@router.post("/start")
def start_demo(req: DemoStartRequest):
    demo_id = f"demo-{uuid.uuid4().hex[:8]}"
    return _init_session(demo_id, req.base_motor_id, req.fault_scenario, req.duration_seconds)


@router.post("/{demo_id}/reset")
def reset_demo(demo_id: str):
    """Re-seeds the same demo session back to a fresh healthy baseline, ready to re-run."""
    session = _demo_sessions.get(demo_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown demo session")
    return _init_session(demo_id, session["base_motor_id"], session["fault_scenario"], session["duration_seconds"])


@router.post("/{demo_id}/step")
def step_demo(demo_id: str):
    """Advances the scripted fault ramp by exactly one reading and rescales the demo history."""
    if demo_id not in _demo_asset_store:
        raise HTTPException(status_code=404, detail="Unknown demo session")

    script = _demo_scripts[demo_id]
    cursor = _demo_cursor[demo_id]
    total_steps = len(script)

    if cursor >= total_steps:
        scored = _demo_scored_cache[demo_id]
        return {
            "step": total_steps,
            "total_steps": total_steps,
            "done": True,
            "health": _sanitize(scored[-1]) if scored else None,
        }

    session = _demo_sessions[demo_id]
    row = dict(script[cursor])
    row["timestamp"] = session["next_timestamp"]
    row["motor_id"] = demo_id
    row["true_condition"] = session["fault_scenario"]
    session["next_timestamp"] = session["next_timestamp"] + pd.Timedelta(seconds=1)

    _demo_asset_store[demo_id] = pd.concat(
        [_demo_asset_store[demo_id], pd.DataFrame([row])], ignore_index=True
    )
    _demo_cursor[demo_id] = cursor + 1

    scored = _pipeline.score_motor_history(_demo_asset_store[demo_id], demo_id)
    _demo_scored_cache[demo_id] = scored

    return {
        "step": cursor + 1,
        "total_steps": total_steps,
        "done": (cursor + 1) >= total_steps,
        "health": _sanitize(scored[-1]) if scored else None,
    }
