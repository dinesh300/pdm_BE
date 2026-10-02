# Motor PdM Intelligence — Backend + Frontend

Spec-driven predictive maintenance system for industrial motors. Combines
ISO 10816-3 / NEMA MG-1 physical thresholds with ML anomaly detection
(Isolation Forest + PyTorch Autoencoder) and probabilistic fault attribution.

## Project layout
```
motor-pdm-intelligence/   <- Python backend (FastAPI)
motor-pdm-frontend/       <- React frontend (Vite)
```

## Running the backend

```bash
cd motor-pdm-intelligence
pip install pandas numpy scikit-learn torch pgmpy fastapi uvicorn shap ruptures

# (Optional) regenerate sample data with embedded fault scenarios
python3 data/raw/generate_sample.py

# Train the Isolation Forest + Autoencoder on healthy baseline data
python3 src/pipeline/train_baseline.py

# Start the API server
python3 -m uvicorn src.api.main:app --reload --port 8000
```

Backend will be live at `http://localhost:8000`. Interactive API docs at
`http://localhost:8000/docs`.

## Running the frontend

```bash
cd motor-pdm-frontend
npm install
npm run dev
```

Frontend will be live at `http://localhost:5173` (Vite default), and expects
the backend running at `http://localhost:8000` (see `src/api/client.js` if
you need to change this).

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/assets` | Fleet overview — every motor with latest health snapshot |
| GET | `/assets/{motor_id}/spec` | Full spec-driven threshold table |
| GET | `/assets/{motor_id}/health` | Current health index, zone classification, likely faults, reasoning |
| GET | `/assets/{motor_id}/history?limit=N` | Full scored history (for charting) |
| POST | `/assets/{motor_id}/ingest` | Ingest one new live reading, get freshly scored result |

## Sample data
5 synthetic motors are pre-loaded on backend startup:
- `MTR-001` — healthy, no fault
- `MTR-002` — gradual bearing wear (vibration + bearing temp co-rising)
- `MTR-003` — overload (current + winding temp rising, RPM sagging)
- `MTR-004` — winding fault / phase imbalance
- `MTR-005` — misalignment (sudden vibration step-change)

## Known limitations (tracked, not silently ignored)
1. **Current imbalance sensitivity**: with independent multi-phase current
   noise, the imbalance metric can drift into "Watch" on a fully healthy
   motor. Zone bands were widened once already (`config/motor_spec.json`);
   may need further tuning against real sensor noise floors.
2. **Fault attribution drift over long history**: the system correctly
   identifies the fault type at the moment of onset, but the specific
   attributed cause can drift later in a motor's life once the onset
   step-change scrolls out of the rolling feature window (rolling-window
   features are autocorrelated, so naive "lock-in" logic risks freezing on
   noise instead of the real event). Needs either a statistically firmer
   trigger or real labeled fault outcomes to train a proper classifier.
   The health index / severity trend itself is unaffected and reliable —
   only the specific fault *label* late in a degradation trend should be
   treated as provisional.

## Next steps
- Swap the in-memory asset store (`src/api/main.py`) for a real DB
- Add authentication before any real deployment
- Feed confirmed AMC fault outcomes back into `fault_priors` to move from
  seeded priors toward a learned Bayesian network (`pgmpy`)
- Extend to Transformer and Pump asset types using the same schema pattern
