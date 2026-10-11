# EduVanta — Federated Learning Model

A fully decoupled Federated Learning system for predicting student academic performance using `SGDRegressor` and Federated Averaging (FedAvg).

---

## Architecture Overview

```
central_server/               client_node/ (runs anywhere)
+------------------+          +----------------------+
|  init_base_model |          |  client_service.py   |
|  server_api.py   |<-------->|  local_trainer.py    |
|  aggregator.py   |  HTTP    |  local_data/*.csv    |
|  weights/*.npz   |          |  local_weights_cache |
+------------------+          +----------------------+
```

**Privacy guarantee**: Raw student records never leave the client. Only model weight vectors (coef + intercept) are transmitted.

---

## Feature Schema (30 features → 1 target)

| Group | Features |
|---|---|
| Academic | current_cgpa, matric_10th, inter_12th, attendance_pct, daily_study_hours, backlogs, assignment_rate |
| Mental/Behavioral | stress_level, self_confidence, time_mgmt, exam_anxiety, motivation_level |
| Lifestyle | sleep_hours, sleep_quality, screen_time_hours, social_media_usage, physical_health, diet_quality, extracurricular_hours |
| Demographic | financial_support, parent_education, region_state, family_income, family_size, internet_access, scholarship |
| Personal | age, gender, region_type, part_time_job |
| **Target** | **next_semester_cgpa** (0.0–10.0) |

**Performance Tiers**: `Excellent` ≥ 8.5 · `Good` ≥ 7.0 · `Average` ≥ 5.0 · `At-Risk` < 5.0

---

## Quick Start

### Prerequisites

```bash
pip install fastapi uvicorn numpy scikit-learn pandas requests pydantic
```

Or install per-component:
```bash
pip install -r central_server/requirements.txt
pip install -r client_node/requirements.txt
```

---

## Step A — Initialise the Central Server Model

```bash
cd central_server
python init_base_model.py
```

**Expected output:**
```
[init] Generating synthetic baseline data (n=2000) ...
[init] Training baseline SGDRegressor ...
[init] Baseline R² on training set: 0.9xxx
[init] Saved weights  -> weights/global_weights_v1.npz
[init] Saved metadata -> weights/feature_metadata.json
[init] Initialisation complete.
```

---

## Step B — Start the Central Server

```bash
cd central_server
uvicorn server_api:app --host 0.0.0.0 --port 8000 --reload
```

Verify it is running:
```bash
curl http://localhost:8000/
```

Browse the interactive API docs:
```
http://localhost:8000/docs
```

### Central Server Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/model/latest-weights` | Download current global weights + metadata |
| POST | `/model/sync-weights` | Receive a client weight update (staged) |
| GET | `/model/buffer-status` | View pending client updates |
| POST | `/model/aggregate` | Run FedAvg → new global version |
| DELETE | `/model/clear-buffer` | Discard staged updates |

---

## Step C — Start the Client Node

Open a **new terminal**:

```bash
cd client_node
uvicorn client_service:app --host 0.0.0.0 --port 8001 --reload
```

**Optional environment variables:**

| Variable | Default | Description |
|---|---|---|
| `EDUVANTA_SERVER_URL` | `http://localhost:8000` | Central server address |
| `EDUVANTA_CLIENT_ID` | `client-<hostname>` | Unique client identifier |
| `EDUVANTA_LOCAL_DATA` | `./local_data` | Path to local CSV records |
| `EDUVANTA_EPOCHS` | `3` | SGD passes per training round |

Example with custom config:
```bash
set EDUVANTA_CLIENT_ID=school-node-01
set EDUVANTA_SERVER_URL=http://192.168.1.10:8000
uvicorn client_service:app --host 0.0.0.0 --port 8001 --reload
```

Browse client API docs:
```
http://localhost:8001/docs
```

---

## Step D — Run a Local Prediction

### Via curl

```bash
curl -X POST http://localhost:8001/local/predict \
  -H "Content-Type: application/json" \
  -d @client_node/sample_student.json
```

### Via PowerShell

```powershell
$body = Get-Content client_node\sample_student.json -Raw
Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8001/local/predict" `
  -ContentType "application/json" `
  -Body $body
```

**Expected response:**
```json
{
  "predicted_cgpa": 7.38,
  "tier": "Good",
  "risk_flag": false,
  "model_version": 1,
  "client_id": "client-HOSTNAME",
  "predicted_at": "2026-10-08T00:00:00Z"
}
```

---

## Step E — Local Incremental Training + Federated Sync

### 1. Add local student records (never sent to server)

Edit `client_node/local_data/local_students.csv` — the file already contains 10 sample records. Add real student rows in the same format (categoricals must be encoded as integers as per the schema).

### 2. Trigger train-and-sync

```bash
curl -X POST http://localhost:8001/local/train-and-sync
```

**What happens internally:**
1. Reads all CSV files from `local_data/`
2. Runs `SGDRegressor.partial_fit()` for `EDUVANTA_EPOCHS` passes
3. POSTs `{client_id, sample_count, coef, intercept, scaler_mean, scaler_scale}` to `/model/sync-weights`

---

## Step F — Aggregate Client Updates (FedAvg Round)

Once one or more clients have synced, trigger aggregation on the server:

```bash
curl -X POST http://localhost:8000/model/aggregate
```

**Expected response:**
```json
{
  "status": "success",
  "new_version": 2,
  "total_samples": 2010,
  "clients_merged": 1,
  "timestamp": "2026-10-08T00:00:00Z"
}
```

A new `weights/global_weights_v2.npz` is saved. All subsequent client `/local/refresh-weights` calls will pull version 2.

---

## Step G — Refresh Client with New Global Weights

```bash
curl -X POST http://localhost:8001/local/refresh-weights
```

---

## Full Federated Round (Summary)

```
1. python central_server/init_base_model.py       # once only
2. uvicorn server_api:app --port 8000             # terminal 1
3. uvicorn client_service:app --port 8001         # terminal 2
4. POST /local/predict                            # test prediction
5. POST /local/train-and-sync                     # client trains + pushes
6. POST /model/aggregate   (server)               # FedAvg -> new version
7. POST /local/refresh-weights                    # client pulls new version
8. Repeat steps 5-7 continuously
```

---

## Directory Structure

```
EduVantaAzMlModel/
├── central_server/
│   ├── init_base_model.py      # Baseline synthetic training + weight export
│   ├── server_api.py           # FastAPI: weight distribution + sync + FedAvg
│   ├── aggregator.py           # Pure FedAvg logic (sample-weighted averaging)
│   ├── requirements.txt
│   └── weights/
│       ├── global_weights_v1.npz
│       ├── global_weights_v2.npz   (after first aggregation)
│       └── feature_metadata.json
│
├── client_node/
│   ├── client_service.py       # FastAPI: predict + train-and-sync endpoints
│   ├── local_trainer.py        # SGD partial_fit + server communication
│   ├── sample_student.json     # Sample prediction payload
│   ├── requirements.txt
│   ├── local_weights_cache.npz # Auto-created: offline fallback cache
│   └── local_data/
│       └── local_students.csv  # Private local records (never transmitted)
│
└── README.md
```

---

## FedAvg Formula

$$W_{global} = \sum_{k=1}^{K} \frac{n_k}{n_{total}} \cdot W_k$$

Where:
- $W_k$ = weight vector from client $k$
- $n_k$ = number of local samples used by client $k$
- $n_{total}$ = sum of all sample counts (including the previous global model)

The existing global model is treated as a "virtual client" with weight `global_sample_count`, ensuring **catastrophic forgetting is prevented** across rounds.

---

## Notes

- **Privacy**: Only `coef`, `intercept`, `scaler_mean`, `scaler_scale`, and `sample_count` leave the client. Raw records stay local.
- **Offline mode**: If the server is unreachable at startup, the client falls back to `local_weights_cache.npz`.
- **Versioning**: Each aggregation creates a new `global_weights_vN.npz`; old versions are retained for rollback.
- **Scalability**: Multiple client nodes can run simultaneously — all POST to the same `/model/sync-weights`, and a single `/model/aggregate` merges all staged updates.
