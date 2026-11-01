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

From the repository root, create and activate one virtual environment:

```powershell
cd D:\Project\EduVantaAzMlModel
py -m venv venv
.\venv\Scripts\Activate.ps1
```

`Activate.ps1` is the PowerShell script that puts the virtual environment's
Python and installed packages first on `PATH`. Activate it separately in every
terminal before starting the server or a client. The three clients do not need
three virtual environments; they use the same environment but different client
IDs and ports.

If PowerShell blocks the script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
```

Install dependencies after activation:

```powershell
pip install -r central_server\requirements.txt
pip install -r client_node\requirements.txt
```

---

## Step A — Initialise the Central Server Model

```powershell
cd D:\Project\EduVantaAzMlModel\central_server
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

```powershell
cd D:\Project\EduVantaAzMlModel\central_server
uvicorn server_api:app --host 0.0.0.0 --port 8000 --reload
```

Verify it is running:
```powershell
Invoke-RestMethod http://localhost:8000/
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

## Step C — Configure and Start Client Nodes

### Client Identity via `json/client_details.json`

Each client node folder contains a `json/client_details.json` file that defines
its identity. This is the **single source of truth** for the client's `client_id`
and `server_url`. You never need to edit Python source files to change identity.

```json
{
  "client_id": "school-node-01",
  "server_url": "http://localhost:8000",
  "client_name": "School Node 01",
  "description": "Client node for school dataset A"
}
```

Environment variables always override the JSON file:

| Variable | JSON field | Default |
|---|---|---|
| `EDUVANTA_CLIENT_ID` | `client_id` | `client-<hostname>` |
| `EDUVANTA_SERVER_URL` | `server_url` | `http://localhost:8000` |
| `EDUVANTA_LOCAL_DATA` | _(env only)_ | `<client_dir>/local_data` |
| `EDUVANTA_WEIGHTS_DIR` | _(env only)_ | `<client_dir>/weights` |
| `EDUVANTA_EPOCHS` | _(env only)_ | `3` |

### Running Multiple Independent Clients on One Machine

Copy the `client_node/` folder for each independent client. Each copy gets its
own `json/client_details.json`, `local_data/`, and `weights/` directory.

```text
Project/
├── client_node/           ← client A (port 8001)
│   ├── json/client_details.json   {"client_id": "school-node-01", ...}
│   ├── local_data/local_students.csv
│   └── weights/
├── client_node_copy/      ← client B (port 8002)
│   ├── json/client_details.json   {"client_id": "school-node-02", ...}
│   ├── local_data/local_students.csv
│   └── weights/
└── another_client/        ← client C (port 8003)
    ├── json/client_details.json   {"client_id": "school-node-03", ...}
    ├── local_data/local_students.csv
    └── weights/
```

Run each in a separate PowerShell terminal. Activate `venv` in each terminal.

### Client A — port 8001

```powershell
cd D:\Project\EduVantaAzMlModel
.\venv\Scripts\Activate.ps1
cd client_node
uvicorn client_service:app --host 0.0.0.0 --port 8001 --reload
```

The `client_id` is read from `client_node/json/client_details.json` automatically.
No environment variables required.

### Client B — port 8002 (copy of client_node)

```powershell
cd D:\Project\EduVantaAzMlModel
.\venv\Scripts\Activate.ps1
cd client_node_copy
uvicorn client_service:app --host 0.0.0.0 --port 8002 --reload
```

`client_node_copy/json/client_details.json` must have a different `client_id`.

### Client C — port 8003 (another copy)

```powershell
cd D:\Project\EduVantaAzMlModel
.\venv\Scripts\Activate.ps1
cd another_client
uvicorn client_service:app --host 0.0.0.0 --port 8003 --reload
```

### Override identity without editing JSON (useful for quick tests)

```powershell
$env:EDUVANTA_CLIENT_ID="school-node-01"
$env:EDUVANTA_SERVER_URL="http://192.168.1.10:8000"
uvicorn client_service:app --host 0.0.0.0 --port 8001 --reload
```

The client saves downloaded global models as `global_weights_vN.npz` in
`weights/`. After each local training round, the locally-adapted weights are
saved as `local_trained_weights.npz` (distinct from the global file) so the
client can restore its state after a restart even when the server is offline.

Browse client API docs:
```
http://localhost:8001/docs
```

---

## Step D — Run a Local Prediction

### Via PowerShell

```powershell
$body = Get-Content client_node\sample_student.json -Raw
Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8001/local/predict" `
  -ContentType "application/json" `
  -Body $body
```

Change `8001` to `8002` or `8003` to call another client.

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

Edit `client_node\local_data\local_students.csv` — the file already contains
10 sample student records. You can use these records for all three demo
clients, or replace/add rows in this file. The client reads every CSV file in
this folder; categorical values must use the encoded integers defined by the
schema.

### 2. Trigger train-and-sync

```powershell
Invoke-RestMethod -Method POST http://localhost:8001/local/train-and-sync
Invoke-RestMethod -Method POST http://localhost:8002/local/train-and-sync
Invoke-RestMethod -Method POST http://localhost:8003/local/train-and-sync
```

**What happens internally:**
1. Reads all CSV files from `client_node\local_data`
2. Runs `SGDRegressor.partial_fit()` for `EDUVANTA_EPOCHS` passes
3. POSTs `{client_id, sample_count, coef, intercept, scaler_mean, scaler_scale}` to `/model/sync-weights`

---

## Step F — Aggregate Client Updates (FedAvg Round)

Once one or more clients have synced, trigger aggregation on the server:

```powershell
Invoke-RestMethod -Method POST http://localhost:8000/model/aggregate
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

```powershell
Invoke-RestMethod -Method POST http://localhost:8001/local/refresh-weights
Invoke-RestMethod -Method POST http://localhost:8002/local/refresh-weights
Invoke-RestMethod -Method POST http://localhost:8003/local/refresh-weights
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
│   ├── init_base_model.py        # Baseline training + weight export (run once)
│   ├── server_api.py             # FastAPI: weight distribution + sync + FedAvg
│   ├── aggregator.py             # Pure FedAvg logic (sample-weighted averaging)
│   ├── requirements.txt
│   ├── json/
│   │   └── api_examples/         # Request/response examples for every endpoint
│   │       ├── health_check.json
│   │       ├── get_latest_weights.json
│   │       ├── sync_weights.json
│   │       ├── buffer_status.json
│   │       ├── aggregate.json
│   │       └── clear_buffer.json
│   └── weights/
│       ├── global_weights_v1.npz
│       ├── global_weights_v2.npz   (created after first aggregation)
│       └── feature_metadata.json
│
├── client_node/                  # Copy this folder for each independent client
│   ├── client_service.py         # FastAPI: predict + train-and-sync endpoints
│   ├── local_trainer.py          # SGD partial_fit + server communication
│   ├── config.py                 # Centralised configuration loader
│   ├── sample_student.json       # Sample prediction payload
│   ├── requirements.txt
│   ├── json/
│   │   ├── client_details.json   # ← edit this to set client_id & server_url
│   │   └── api_examples/         # Request/response examples for every endpoint
│   │       ├── health_check.json
│   │       ├── local_status.json
│   │       ├── predict.json
│   │       ├── train_and_sync.json
│   │       └── refresh_weights.json
│   ├── local_data/
│   │   └── local_students.csv    # Private local records (never transmitted)
│   └── weights/                  # ALL weight files live here (single source of truth)
│       ├── global_weights_v1.npz         # Downloaded from server
│       ├── local_weights_cache.npz       # Offline fallback cache
│       ├── local_weights_cache_meta.json # Metadata for the cache
│       ├── local_trained_weights.npz     # Saved after each local training round
│       ├── local_trained_weights_meta.json
│       └── feature_metadata.json
│
├── API_TESTING.md                # Complete endpoint testing guide
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
- **Offline mode**: If the server is unreachable at startup, the client falls back to `local_trained_weights.npz` (post-training state), then `global_weights_vN.npz`, then `local_weights_cache.npz`.
- **Versioning**: Each aggregation creates a new `global_weights_vN.npz`; old versions are retained for rollback.
- **Scalability**: Multiple client nodes can run simultaneously — all POST to the same `/model/sync-weights`, and a single `/model/aggregate` merges all staged updates.
- **Multi-instance**: Copy `client_node/` for each independent instance; edit its `json/client_details.json`. No Python source code changes are needed.
- **Configuration**: `json/client_details.json` sets `client_id` and `server_url`. Environment variables always take priority over the JSON file.
- **Weight isolation**: Every copied client folder has its own `weights/` directory. Clients never share or overwrite each other's weight files.
- **Local persistence**: After each training round, `local_trained_weights.npz` is saved so the locally-adapted model survives service restarts.
