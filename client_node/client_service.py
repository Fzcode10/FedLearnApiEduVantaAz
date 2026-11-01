"""
client_service.py
=================
EduVanta Client Node — FastAPI service.

Endpoints
---------
GET  /                          Health check
GET  /local/status              Model version, server URL, local sample count
POST /local/predict             Predict next_semester_cgpa for a student
POST /local/train-and-sync      Incremental train on local data -> push to server
POST /local/refresh-weights     Force re-download of global weights from server

Configuration via environment variables (with defaults):
    EDUVANTA_SERVER_URL   = http://localhost:8000
    EDUVANTA_CLIENT_ID    = client-<hostname>
    EDUVANTA_LOCAL_DATA   = <this file's directory>/local_data
    EDUVANTA_WEIGHTS_DIR  = <this file's directory>/weights
    EDUVANTA_EPOCHS       = 3

Start:
    uvicorn client_service:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import load_client_config
from local_trainer import (
    LocalModel,
    fetch_global_weights,
    latest_versioned_weights,
    train_and_sync,
    version_from_weights_file,
)

# ── configuration ──────────────────────────────────────────────────────────────
config = load_client_config()

# ── global model state ─────────────────────────────────────────────────────────
_model_lock  = threading.Lock()
_local_model = LocalModel()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EduVanta Client Node",
    description="Local prediction and federated learning sync service for EduVanta.",
    version="1.0.0",
)


# ── startup: fetch weights ─────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    print(f"[client] Starting EduVanta Client Node")
    print(f"[client] client_id    = {config.client_id}")
    if config.client_name:
        print(f"[client] client_name  = {config.client_name}")
    print(f"[client] server_url   = {config.server_url}")
    print(f"[client] local_data   = {config.local_data_dir}")
    try:
        with _model_lock:
            fetch_global_weights(
                config.server_url,
                _local_model,
                cache_path=config.cache_path,
                weights_dir=config.weights_dir,
            )
        print(f"[client] Model ready: v{_local_model.version}")
    except Exception as exc:
        print(f"[client] WARNING: Could not load model on startup: {exc}")
        # Last-resort fallback: try locally-trained weights
        local_trained_path = os.path.join(
            config.weights_dir, "local_trained_weights.npz"
        )
        local_trained_meta = os.path.join(
            config.weights_dir, "local_trained_weights_meta.json"
        )
        if os.path.exists(local_trained_path):
            try:
                with _model_lock:
                    _local_model.load_from_versioned_file(
                        local_trained_path,
                        metadata_path=local_trained_meta
                        if os.path.exists(local_trained_meta)
                        else os.path.join(config.weights_dir, "feature_metadata.json"),
                    )
                print(
                    f"[client] Loaded local trained weights fallback: "
                    f"v{_local_model.version}"
                )
                return
            except Exception as fallback_exc:
                print(
                    f"[client] WARNING: Could not load local trained "
                    f"weights either: {fallback_exc}"
                )
        print("[client] Use POST /local/refresh-weights once the server is available.")


# ── request/response schemas ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    # Academic
    current_cgpa:        float = Field(..., ge=0.0, le=10.0)
    matric_10th:         float = Field(..., ge=0.0, le=100.0)
    inter_12th:          float = Field(..., ge=0.0, le=100.0)
    attendance_pct:      float = Field(..., ge=0.0, le=100.0)
    daily_study_hours:   float = Field(..., ge=0.0, le=24.0)
    backlogs:            int   = Field(..., ge=0)
    assignment_rate:     float = Field(..., ge=0.0, le=100.0)
    # Mental / Behavioral (0-10 scale)
    stress_level:        float = Field(..., ge=0.0, le=10.0)
    self_confidence:     float = Field(..., ge=0.0, le=10.0)
    time_mgmt:           float = Field(..., ge=0.0, le=10.0)
    exam_anxiety:        float = Field(..., ge=0.0, le=10.0)
    motivation_level:    float = Field(..., ge=0.0, le=10.0)
    # Lifestyle
    sleep_hours:         float = Field(..., ge=0.0, le=24.0)
    sleep_quality:       float = Field(..., ge=0.0, le=10.0)
    screen_time_hours:   float = Field(..., ge=0.0, le=24.0)
    social_media_usage:  float = Field(..., ge=0.0, le=10.0)
    physical_health:     float = Field(..., ge=0.0, le=10.0)
    diet_quality:        float = Field(..., ge=0.0, le=10.0)
    extracurricular_hours: float = Field(..., ge=0.0, le=24.0)
    # Demographic
    financial_support:   float = Field(..., ge=0.0, le=10.0)
    parent_education:    str   = Field(..., description="Illiterate | School | Graduate | PostGraduate")
    region_state:        str   = Field(..., description="Bihar | UP | Rajasthan | Kerala | West Bengal | Other")
    family_income:       str   = Field(..., description="Low | Medium | High")
    family_size:         int   = Field(..., ge=1)
    internet_access:     int   = Field(..., ge=0, le=1)
    scholarship:         int   = Field(..., ge=0, le=1)
    # Personal
    age:                 int   = Field(..., ge=14, le=35)
    gender:              str   = Field(..., description="Male | Female | Other")
    region_type:         str   = Field(..., description="Rural | Urban")
    part_time_job:       int   = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_cgpa: float
    tier:           str
    risk_flag:      bool
    model_version:  int
    client_id:      str
    predicted_at:   str


class TrainSyncResponse(BaseModel):
    status:          str
    samples_trained: int
    client_id:       str
    server_response: Dict[str, Any]
    completed_at:    str


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "service":      "EduVanta Client Node",
        "client_id":    config.client_id,
        "server_url":   config.server_url,
        "model_version": _local_model.version,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


@app.get("/local/status", tags=["Local"])
def status():
    """Returns information about the local model and configuration."""
    return {
        "client_id":      config.client_id,
        "server_url":     config.server_url,
        "model_version":  _local_model.version,
        "sample_count":   _local_model.sample_count,
        "feature_count":  len(_local_model.feature_columns),
        "local_data_dir": config.local_data_dir,
        "epochs":         config.epochs,
    }


@app.post("/local/predict", response_model=PredictResponse, tags=["Local"])
def predict(req: PredictRequest):
    """
    Predict next-semester CGPA for a student using the local model.
    Returns predicted_cgpa, performance tier, and risk_flag.
    """
    try:
        with _model_lock:
            latest_path = latest_versioned_weights(config.weights_dir)
            if latest_path:
                latest_version = version_from_weights_file(latest_path)
                if latest_version > _local_model.version:
                    _local_model.load_from_versioned_file(
                        latest_path,
                        metadata_path=os.path.join(
                            config.weights_dir, "feature_metadata.json"
                        ),
                    )

            if _local_model.version == 0:
                raise RuntimeError(
                    "Model not loaded. Call POST /local/refresh-weights first."
                )

            features = req.model_dump()
            result = _local_model.predict(features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return PredictResponse(
        predicted_cgpa=result["predicted_cgpa"],
        tier=result["tier"],
        risk_flag=result["risk_flag"],
        model_version=result["model_version"],
        client_id=config.client_id,
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/local/train-and-sync", response_model=TrainSyncResponse, tags=["Local"])
def train_and_sync_endpoint():
    """
    1. Reads all CSV files from local_data/.
    2. Runs incremental SGD partial_fit (EPOCHS passes).
    3. POSTs resulting weight deltas to the central server.

    The local model is updated in-place after training.
    """
    if _local_model.version == 0:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call POST /local/refresh-weights first."
        )

    try:
        with _model_lock:
            server_resp = train_and_sync(
                server_url=config.server_url,
                model=_local_model,
                local_data_dir=config.local_data_dir,
                client_id=config.client_id,
                epochs=config.epochs,
                weights_dir=config.weights_dir,
            )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sync failed: {e}")

    return TrainSyncResponse(
        status="synced",
        samples_trained=_local_model.sample_count,
        client_id=config.client_id,
        server_response=server_resp,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/local/refresh-weights", tags=["Local"])
def refresh_weights():
    """Force re-download of the latest global weights from the central server."""
    try:
        with _model_lock:
            fetch_global_weights(
                config.server_url,
                _local_model,
                cache_path=config.cache_path,
                weights_dir=config.weights_dir,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch weights: {e}")

    return {
        "status":        "refreshed",
        "model_version": _local_model.version,
        "sample_count":  _local_model.sample_count,
        "refreshed_at":  datetime.now(timezone.utc).isoformat(),
    }
