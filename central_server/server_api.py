"""
server_api.py
=============
Central EduVanta Server — FastAPI service.

Endpoints
---------
GET  /                          Health check
GET  /model/latest-weights      Returns current global model weights + metadata
POST /model/sync-weights        Receives a client weight update (adds to staging buffer)
GET  /model/buffer-status       Shows pending updates in the staging buffer
POST /model/aggregate           Runs FedAvg on staged updates -> new global version
DELETE /model/clear-buffer      Empties the staging buffer without aggregating

Start:
    uvicorn server_api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import glob
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from aggregator import federated_average

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
METADATA_FILE = os.path.join(WEIGHTS_DIR, "feature_metadata.json")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_weights_path() -> str:
    """Return path of the highest-version global_weights_vN.npz file."""
    pattern = os.path.join(WEIGHTS_DIR, "global_weights_v*.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            "No global weights found. Run central_server/init_base_model.py first."
        )
    # Sort by version number extracted from filename
    files.sort(key=lambda p: int(os.path.basename(p).replace("global_weights_v", "").replace(".npz", "")))
    return files[-1]


def _load_weights(path: str) -> Dict[str, Any]:
    """Load a .npz weights file and return a plain dict."""
    data = np.load(path, allow_pickle=False)
    return {
        "coef":         data["coef"],
        "intercept":    data["intercept"],
        "scaler_mean":  data["scaler_mean"],
        "scaler_scale": data["scaler_scale"],
        "version":      int(data["version"]),
        "sample_count": int(data["sample_count"]),
    }


def _save_weights(version: int, result: Dict[str, np.ndarray]) -> str:
    """Persist aggregated weights to a new versioned .npz file."""
    path = os.path.join(WEIGHTS_DIR, f"global_weights_v{version}.npz")
    np.savez(
        path,
        coef=result["coef"],
        intercept=result["intercept"],
        scaler_mean=result["scaler_mean"],
        scaler_scale=result["scaler_scale"],
        version=np.array(version),
        sample_count=np.array(result["total_samples"]),
    )
    return path


def _load_metadata() -> Dict[str, Any]:
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("feature_metadata.json not found. Run init_base_model.py first.")
    with open(METADATA_FILE) as f:
        return json.load(f)


# ── application state ──────────────────────────────────────────────────────────
# Thread-safe staging buffer for incoming client updates
_buffer_lock   = threading.Lock()
_staging_buffer: List[Dict[str, Any]] = []

app = FastAPI(
    title="EduVanta Central Model Server",
    description="Federated Learning coordination server for EduVanta student performance model.",
    version="1.0.0",
)


# ── request/response schemas ───────────────────────────────────────────────────

class SyncWeightsRequest(BaseModel):
    client_id:    str          = Field(..., description="Unique identifier for the client node")
    sample_count: int          = Field(..., ge=1, description="Number of local samples used in this training round")
    coef:         List[float]  = Field(..., description="Model coefficient vector")
    intercept:    List[float]  = Field(..., description="Model intercept (single-element list)")
    scaler_mean:  Optional[List[float]] = Field(None, description="StandardScaler mean (optional)")
    scaler_scale: Optional[List[float]] = Field(None, description="StandardScaler scale (optional)")


class AggregateResponse(BaseModel):
    status:      str
    new_version: int
    total_samples: int
    clients_merged: int
    timestamp:   str


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "EduVanta Central Model Server", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model/latest-weights", tags=["Model"])
def get_latest_weights():
    """
    Returns the current global model weights, scaler parameters,
    model version, and feature metadata as JSON.
    """
    try:
        path = _latest_weights_path()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    w  = _load_weights(path)
    try:
        meta = _load_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "version":         w["version"],
        "sample_count":    w["sample_count"],
        "coef":            w["coef"].tolist(),
        "intercept":       w["intercept"].tolist(),
        "scaler_mean":     w["scaler_mean"].tolist(),
        "scaler_scale":    w["scaler_scale"].tolist(),
        "feature_columns": meta["feature_columns"],
        "categorical_maps": meta["categorical_maps"],
        "performance_tiers": meta.get("performance_tiers", {}),
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
    }


@app.post("/model/sync-weights", tags=["Model"], status_code=202)
def sync_weights(payload: SyncWeightsRequest):
    """
    Receives a client weight update and adds it to the staging buffer.
    The buffer is consumed when POST /model/aggregate is called.
    """
    entry = {
        "client_id":    payload.client_id,
        "sample_count": payload.sample_count,
        "coef":         np.asarray(payload.coef,      dtype=np.float64),
        "intercept":    np.asarray(payload.intercept, dtype=np.float64),
        "scaler_mean":  np.asarray(payload.scaler_mean,  dtype=np.float64) if payload.scaler_mean  else None,
        "scaler_scale": np.asarray(payload.scaler_scale, dtype=np.float64) if payload.scaler_scale else None,
        "received_at":  datetime.now(timezone.utc).isoformat(),
    }
    with _buffer_lock:
        _staging_buffer.append(entry)
        buffer_size = len(_staging_buffer)

    print(f"[server] Received update from '{payload.client_id}' "
          f"(samples={payload.sample_count}). Buffer size: {buffer_size}")

    return {
        "status":   "accepted",
        "client_id": payload.client_id,
        "buffer_size": buffer_size,
        "message": "Update staged. Call POST /model/aggregate to run FedAvg.",
    }


@app.get("/model/buffer-status", tags=["Model"])
def buffer_status():
    """Shows how many client updates are pending aggregation."""
    with _buffer_lock:
        count = len(_staging_buffer)
        clients = [e["client_id"] for e in _staging_buffer]
    return {"pending_updates": count, "client_ids": clients}


@app.post("/model/aggregate", tags=["Model"], response_model=AggregateResponse)
def aggregate():
    """
    Runs Federated Averaging over all staged client updates.
    Produces a new versioned global_weights_vN.npz and clears the buffer.
    """
    with _buffer_lock:
        if not _staging_buffer:
            raise HTTPException(
                status_code=400,
                detail="No pending client updates in buffer. Nothing to aggregate."
            )
        updates_snapshot = list(_staging_buffer)

    # Load current global state
    try:
        path = _latest_weights_path()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    current = _load_weights(path)

    # Run FedAvg
    result = federated_average(
        client_updates=updates_snapshot,
        global_coef=current["coef"],
        global_intercept=current["intercept"],
        global_scaler_mean=current["scaler_mean"],
        global_scaler_scale=current["scaler_scale"],
        global_sample_count=current["sample_count"],
    )

    new_version = current["version"] + 1
    new_path    = _save_weights(new_version, result)

    # Update metadata version
    try:
        meta = _load_metadata()
        meta["version"] = new_version
        with open(METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass  # Non-fatal

    # Clear buffer
    with _buffer_lock:
        _staging_buffer.clear()

    print(f"[server] Aggregation complete -> version {new_version}, "
          f"saved to {new_path}")

    return AggregateResponse(
        status="success",
        new_version=new_version,
        total_samples=int(result["total_samples"]),
        clients_merged=len(updates_snapshot),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.delete("/model/clear-buffer", tags=["Model"])
def clear_buffer():
    """Empties the staging buffer without running aggregation."""
    with _buffer_lock:
        count = len(_staging_buffer)
        _staging_buffer.clear()
    return {"status": "cleared", "discarded_updates": count}
