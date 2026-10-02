"""
FastAPI serving layer for the Motor PdM intelligence engine.
Endpoints are designed around what the React frontend needs:
fleet overview, per-asset spec table, per-asset live health, and history
for charting.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hashlib
import json
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import pandas as pd

from pipeline.live_feed_inference import MotorInferencePipeline
from . import demo_routes

app = FastAPI(title="Motor PdM Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = MotorInferencePipeline()
demo_routes.attach(pipeline)
app.include_router(demo_routes.router)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_motor_data.csv"

# In-memory store: {motor_id: raw_dataframe}. Swap for a real DB later —
# this is the seam where PreventiveGrid's existing backend would plug in.
_asset_store: dict[str, pd.DataFrame] = {}
_scored_cache: dict[str, list] = {}
_explanation_cache: dict[str, dict] = {}  # motor_id -> {"key": str, "text": str}

EXPLAIN_SYSTEM_PROMPT = (
    "You are a maintenance engineer explaining a motor's spec-driven predictive-"
    "maintenance diagnosis to a plant operator who wants the short version, not "
    "a report. You will be given structured JSON facts about one motor: rated "
    "nameplate values, spec zone classification per parameter, the fused health "
    "index and its contributing scores, the ranked likely fault causes, and a "
    "trend table comparing each sensor's early-baseline value to its latest "
    "reading (including per-phase current under phase_trends, labeled Phase R / "
    "Phase Y / Phase B).\n\n"
    "Rules:\n"
    "- Only state facts directly supported by the provided JSON. Never invent "
    "sensor values, thresholds, dates, phase behavior, or causes not present in "
    "the data.\n"
    "- Use the trend table to say concretely which parameter(s) or phase(s) moved "
    "and which stayed flat -- compare baseline vs latest yourself and describe "
    "the real pattern (e.g. one phase climbing while the other two stay flat, "
    "or a temperature creeping up together with vibration). Do not describe a "
    "change as sudden unless the baseline-to-latest numbers actually show a "
    "large jump.\n"
    "- Connect the pattern to the top likely fault only if the signature "
    "genuinely matches; otherwise just describe what's happening physically.\n"
    "- Be brief: 2-4 short sentences, plain language, flowing prose -- no "
    "markdown, no headers, no bullet lists, no jargon that isn't explained.\n"
    "- If every parameter is in its normal zone and no trend has moved "
    "meaningfully, say so plainly and do not manufacture a fault narrative."
)

TREND_BASELINE_WINDOW = 20  # leading readings assumed to reflect healthy commissioning


def _load_sample_data():
    raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    for motor_id, group in raw.groupby("motor_id"):
        _asset_store[motor_id] = group.sort_values("timestamp").reset_index(drop=True)


import numpy as np


def _sanitize(obj):
    """Recursively converts numpy scalar types to native Python types for JSON serialization."""
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


class ReadingIn(BaseModel):
    timestamp: str
    current_A: float
    current_A_ph2: float
    current_A_ph3: float
    voltage_V: float
    rpm: float
    vibration_mms: float
    winding_temp_C: float
    bearing_temp_C: float
    power_factor: float


@app.on_event("startup")
def startup():
    _load_sample_data()


@app.get("/assets")
def list_assets():
    """Fleet overview: every asset with its latest health snapshot."""
    results = []
    for motor_id, df in _asset_store.items():
        scored = _scored_cache.get(motor_id)
        if scored is None:
            scored = pipeline.score_motor_history(df, motor_id)
            _scored_cache[motor_id] = scored
        latest = scored[-1] if scored else None
        asset = pipeline.asset_registry.get(motor_id, {})
        results.append({
            "motor_id": motor_id,
            "power_kw": asset.get("power_kw"),
            "reading_count": len(df),
            "latest_health": latest,
        })
    return _sanitize(results)


@app.get("/assets/{motor_id}/spec")
def get_asset_spec(motor_id: str):
    """Spec-driven threshold table for frontend display."""
    if motor_id not in _asset_store:
        raise HTTPException(status_code=404, detail="Asset not found")
    return _sanitize(pipeline.get_spec_table(motor_id))


@app.get("/assets/{motor_id}/health")
def get_asset_health(motor_id: str):
    """Current (latest) health index, zone classification, faults, reasoning."""
    if motor_id not in _asset_store:
        raise HTTPException(status_code=404, detail="Asset not found")
    scored = _scored_cache.get(motor_id)
    if scored is None:
        scored = pipeline.score_motor_history(_asset_store[motor_id], motor_id)
        _scored_cache[motor_id] = scored
    if not scored:
        raise HTTPException(status_code=400, detail="Not enough history to score yet")
    return _sanitize(scored[-1])


@app.post("/assets/{motor_id}/explain")
def explain_diagnosis(motor_id: str):
    """
    On-demand narrative explanation of the motor's current diagnosis, generated
    by an LLM call grounded strictly in this motor's own structured data (spec,
    health, contributing scores, likely faults). Cached per motor and only
    regenerated when the underlying diagnosis actually changes.
    """
    if motor_id not in _asset_store:
        raise HTTPException(status_code=404, detail="Asset not found")

    scored = _scored_cache.get(motor_id)
    if scored is None:
        scored = pipeline.score_motor_history(_asset_store[motor_id], motor_id)
        _scored_cache[motor_id] = scored
    if not scored:
        raise HTTPException(status_code=400, detail="Not enough history to score yet")

    health = _sanitize(scored[-1])
    spec = _sanitize(pipeline.get_spec_table(motor_id))

    raw_df = _asset_store[motor_id]
    baseline_df = raw_df.iloc[: min(TREND_BASELINE_WINDOW, len(raw_df))]
    latest_row = raw_df.iloc[-1]

    def _trend(col):
        return {
            "baseline": round(float(baseline_df[col].mean()), 2),
            "latest": round(float(latest_row[col]), 2),
        }

    sensor_trends = {
        col: _trend(col)
        for col in ["vibration_mms", "winding_temp_C", "bearing_temp_C", "current_imbalance_pct"]
        if col in raw_df.columns
    }
    phase_labels = {"current_A": "Phase R", "current_A_ph2": "Phase Y", "current_A_ph3": "Phase B"}
    phase_trends = {
        label: _trend(col) for col, label in phase_labels.items() if col in raw_df.columns
    }

    facts = {
        "motor_id": motor_id,
        "power_kw": spec.get("power_kw"),
        "rated_current_A": spec.get("rated_current_A"),
        "rated_voltage_V": spec.get("rated_voltage_V"),
        "rated_rpm": spec.get("rated_rpm"),
        "health_index": health["health_index"],
        "severity": health["severity"],
        "worst_spec_zone": health["worst_spec_zone"],
        "zone_classification": health["zone_classification"],
        "contributing_scores": health["contributing_scores"],
        "likely_faults": health["likely_faults"],
        "reasoning": health["reasoning"],
        "sensor_trends": sensor_trends,
        "phase_trends": phase_trends,
    }
    facts_json = json.dumps(facts, sort_keys=True)
    cache_key = hashlib.sha256(facts_json.encode()).hexdigest()

    cached = _explanation_cache.get(motor_id)
    if cached and cached["key"] == cache_key:
        return {"explanation": cached["text"], "cached": True}

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set on the server. Set it as an "
                   "environment variable and restart the backend to enable AI explanations.",
        )

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=400,
            system=EXPLAIN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": facts_json}],
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"AI explanation request failed: {e.message}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    _explanation_cache[motor_id] = {"key": cache_key, "text": text}
    return {"explanation": text, "cached": False}


@app.get("/assets/{motor_id}/history")
def get_asset_history(motor_id: str, limit: Optional[int] = 200):
    """Full scored history for frontend charting (health index over time, etc)."""
    if motor_id not in _asset_store:
        raise HTTPException(status_code=404, detail="Asset not found")
    scored = _scored_cache.get(motor_id)
    if scored is None:
        scored = pipeline.score_motor_history(_asset_store[motor_id], motor_id)
        _scored_cache[motor_id] = scored
    return _sanitize(scored[-limit:])


@app.post("/assets/{motor_id}/ingest")
def ingest_reading(motor_id: str, reading: ReadingIn):
    """
    Ingest a single new live reading, append to the asset's history,
    and return the freshly scored result for just this reading.
    This is the seam a real sensor feed / MQTT bridge would call.
    """
    row = reading.dict()
    row["timestamp"] = pd.to_datetime(row["timestamp"])
    row["motor_id"] = motor_id
    row["current_imbalance_pct"] = (
        max(row["current_A"], row["current_A_ph2"], row["current_A_ph3"]) -
        min(row["current_A"], row["current_A_ph2"], row["current_A_ph3"])
    ) / ((row["current_A"] + row["current_A_ph2"] + row["current_A_ph3"]) / 3) * 100
    row["true_condition"] = None

    if motor_id not in _asset_store:
        _asset_store[motor_id] = pd.DataFrame([row])
    else:
        _asset_store[motor_id] = pd.concat(
            [_asset_store[motor_id], pd.DataFrame([row])], ignore_index=True
        )

    scored = pipeline.score_motor_history(_asset_store[motor_id], motor_id)
    _scored_cache[motor_id] = scored
    if not scored:
        return {"status": "accepted", "message": "Reading stored; not enough history to score yet."}
    return _sanitize(scored[-1])


@app.get("/")
def root():
    return {"status": "ok", "service": "Motor PdM Intelligence API"}
