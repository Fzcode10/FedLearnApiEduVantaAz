"""
local_trainer.py
================
Handles all local model operations for the EduVanta client node:

1. fetch_global_weights(server_url)
   Downloads the latest global weights from the central server and
   initialises a local SGDRegressor + StandardScaler.

2. predict(features_dict)
   Runs offline inference using the locally cached model weights.
   Returns predicted_cgpa, tier, and risk_flag.

3. train_and_sync(server_url, local_data_dir)
   Reads all CSV records from local_data/, performs incremental
   SGD partial_fit passes, then POSTs the updated weights back to
   the server's /model/sync-weights endpoint.
"""

from __future__ import annotations
import glob
import json
import os
import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# ── tier helper ────────────────────────────────────────────────────────────────
def cgpa_to_tier(cgpa: float) -> Tuple[str, bool]:
    """Map a predicted CGPA value to a performance tier and risk flag."""
    if cgpa >= 8.5:
        return "Excellent", False
    elif cgpa >= 7.0:
        return "Good", False
    elif cgpa >= 5.0:
        return "Average", False
    else:
        return "At-Risk", True


# ── LocalModel ─────────────────────────────────────────────────────────────────
class LocalModel:
    """
    Wraps a local SGDRegressor + StandardScaler loaded from server weights.
    Thread-safe for read operations; update operations should use the provided
    lock in client_service.py.
    """

    def __init__(self):
        self.coef:          Optional[np.ndarray] = None
        self.intercept:     Optional[np.ndarray] = None
        self.scaler_mean:   Optional[np.ndarray] = None
        self.scaler_scale:  Optional[np.ndarray] = None
        self.version:       int = 0
        self.sample_count:  int = 0
        self.feature_columns: list[str] = []
        self.categorical_maps: Dict[str, Dict[str, int]] = {}
        self._model:   Optional[SGDRegressor]  = None
        self._scaler:  Optional[StandardScaler] = None

    # ── initialisation ─────────────────────────────────────────────────────────
    def _rebuild_sklearn_objects(self):
        """Reconstruct scikit-learn objects from stored numpy arrays."""
        n_features = len(self.coef)

        self._scaler = StandardScaler()
        self._scaler.mean_  = self.scaler_mean.copy()
        self._scaler.scale_ = self.scaler_scale.copy()
        self._scaler.var_   = self.scaler_scale ** 2
        self._scaler.n_features_in_ = n_features
        self._scaler.n_samples_seen_ = np.array(self.sample_count)

        self._model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.001,
            learning_rate="invscaling",
            eta0=0.01,
            random_state=42,
        )
        # Manually inject weights so the model is ready without fitting
        self._model.coef_      = self.coef.copy()
        self._model.intercept_ = self.intercept.copy()
        self._model.t_         = float(self.sample_count)
        self._model.n_features_in_ = n_features

    def load_from_server_response(self, data: Dict[str, Any]):
        """Populate local model from a /model/latest-weights JSON response."""
        self.coef         = np.asarray(data["coef"],         dtype=np.float64)
        self.intercept    = np.asarray(data["intercept"],    dtype=np.float64)
        self.scaler_mean  = np.asarray(data["scaler_mean"],  dtype=np.float64)
        self.scaler_scale = np.asarray(data["scaler_scale"], dtype=np.float64)
        self.version      = int(data["version"])
        self.sample_count = int(data["sample_count"])
        self.feature_columns  = data["feature_columns"]
        self.categorical_maps = data.get("categorical_maps", {})
        self._rebuild_sklearn_objects()
        print(f"[trainer] Loaded global model v{self.version} "
              f"(sample_count={self.sample_count}, features={len(self.feature_columns)})")

    def load_from_cache(self, cache_path: str):
        """Load weights from a local .npz cache file (offline fallback)."""
        data = np.load(cache_path, allow_pickle=False)
        self.coef         = data["coef"]
        self.intercept    = data["intercept"]
        self.scaler_mean  = data["scaler_mean"]
        self.scaler_scale = data["scaler_scale"]
        self.version      = int(data["version"])
        self.sample_count = int(data["sample_count"])
        meta_path = cache_path.replace(".npz", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_columns  = meta.get("feature_columns", [])
            self.categorical_maps = meta.get("categorical_maps", {})
        self._rebuild_sklearn_objects()
        print(f"[trainer] Loaded cached model v{self.version} from {cache_path}")

    def save_to_cache(self, cache_path: str):
        """Persist current weights to a local .npz cache file."""
        np.savez(
            cache_path,
            coef=self.coef,
            intercept=self.intercept,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            version=np.array(self.version),
            sample_count=np.array(self.sample_count),
        )
        meta_path = cache_path.replace(".npz", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_columns": self.feature_columns,
                "categorical_maps": self.categorical_maps,
            }, f, indent=2)
        print(f"[trainer] Weights cached to {cache_path}")

    # ── feature encoding ───────────────────────────────────────────────────────
    def _encode_input(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert a raw feature dict (with string categorical values) into
        the ordered float64 vector expected by the model.
        """
        row = []
        for col in self.feature_columns:
            val = features.get(col)
            if val is None:
                raise ValueError(f"Missing required feature: '{col}'")
            # Encode categoricals using the stored mapping
            if col in self.categorical_maps:
                mapping = self.categorical_maps[col]
                if isinstance(val, str):
                    if val not in mapping:
                        raise ValueError(
                            f"Unknown category '{val}' for '{col}'. "
                            f"Valid: {list(mapping.keys())}"
                        )
                    val = mapping[val]
            row.append(float(val))
        return np.asarray(row, dtype=np.float64).reshape(1, -1)

    # ── prediction ─────────────────────────────────────────────────────────────
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict next_semester_cgpa from a raw feature dict.

        Returns
        -------
        dict with keys:
          predicted_cgpa (float), tier (str), risk_flag (bool),
          model_version (int)
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not initialised. Call fetch_global_weights first.")

        X_raw    = self._encode_input(features)
        X_scaled = self._scaler.transform(X_raw)
        raw_pred = float(self._model.predict(X_scaled)[0])
        cgpa     = float(np.clip(raw_pred, 0.0, 10.0))
        tier, risk_flag = cgpa_to_tier(cgpa)

        return {
            "predicted_cgpa": round(cgpa, 4),
            "tier":           tier,
            "risk_flag":      risk_flag,
            "model_version":  self.version,
        }

    # ── incremental local training ─────────────────────────────────────────────
    def partial_fit_on_local_data(
        self,
        local_data_dir: str,
        epochs: int = 3,
    ) -> int:
        """
        Read all CSV files from local_data_dir, encode them, scale them
        (re-using the server's scaler params — no re-fitting the scaler),
        and run SGD partial_fit passes.

        Returns the number of samples trained on.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not initialised.")

        csv_files = glob.glob(os.path.join(local_data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {local_data_dir}. "
                "Add student records before training."
            )

        frames = [pd.read_csv(f) for f in csv_files]
        df     = pd.concat(frames, ignore_index=True)

        # Encode categoricals
        for col, mapping in self.categorical_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        required_cols = self.feature_columns + ["next_semester_cgpa"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Local CSV missing columns: {missing}")

        df = df.dropna(subset=required_cols)
        if df.empty:
            raise ValueError("No valid rows after dropping NaNs.")

        X = df[self.feature_columns].values.astype(np.float64)
        y = df["next_semester_cgpa"].values.astype(np.float64)

        # Scale using existing server scaler (no re-fit)
        X_scaled = self._scaler.transform(X)

        n_samples = len(X_scaled)
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n_samples)
            self._model.partial_fit(X_scaled[idx], y[idx])
            print(f"[trainer] Epoch {epoch}/{epochs} complete ({n_samples} samples)")

        # Sync back to numpy arrays
        self.coef      = self._model.coef_.copy()
        self.intercept = self._model.intercept_.copy()
        self.sample_count = n_samples
        print(f"[trainer] Local training done: {n_samples} samples, {epochs} epochs.")
        return n_samples


# ── module-level convenience functions ─────────────────────────────────────────

_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_weights_cache.npz")


def fetch_global_weights(
    server_url: str,
    model: LocalModel,
    cache_path: str = _DEFAULT_CACHE,
    timeout: int = 10,
) -> LocalModel:
    """
    Download the latest weights from the central server.
    Falls back to the local cache if the server is unreachable.
    """
    endpoint = server_url.rstrip("/") + "/model/latest-weights"
    try:
        resp = requests.get(endpoint, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        model.load_from_server_response(data)
        model.save_to_cache(cache_path)
        return model
    except Exception as exc:
        print(f"[trainer] Could not reach server ({exc}). Trying local cache ...")
        if os.path.exists(cache_path):
            model.load_from_cache(cache_path)
            return model
        raise RuntimeError(
            "Server unreachable and no local cache found. "
            "Start the central server or provide a cache file."
        ) from exc


def train_and_sync(
    server_url: str,
    model: LocalModel,
    local_data_dir: str,
    client_id: str,
    epochs: int = 3,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Convenience function: partial_fit on local data, then POST weights to server.

    Returns the server's response JSON.
    """
    n_samples = model.partial_fit_on_local_data(local_data_dir, epochs=epochs)

    payload = {
        "client_id":    client_id,
        "sample_count": n_samples,
        "coef":         model.coef.tolist(),
        "intercept":    model.intercept.tolist(),
        "scaler_mean":  model.scaler_mean.tolist(),
        "scaler_scale": model.scaler_scale.tolist(),
    }

    endpoint = server_url.rstrip("/") + "/model/sync-weights"
    print(f"[trainer] Pushing {n_samples} samples to {endpoint} ...")
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    print(f"[trainer] Sync response: {result}")
    return result
