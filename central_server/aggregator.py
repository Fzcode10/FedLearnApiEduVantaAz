"""
aggregator.py
=============
FedAvg implementation.

Receives a list of client weight updates, each with:
  - coef         : np.ndarray  (model coefficients)
  - intercept    : np.ndarray  (model intercept)
  - sample_count : int         (number of local samples used in training)

Produces a new global weight vector as:
    W_global = sum( (n_k / n_total) * W_k )   for all k clients

Also re-computes the merged scaler parameters as a weighted average so
the server can always export a complete, self-contained .npz snapshot.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any


def federated_average(
    client_updates: List[Dict[str, Any]],
    global_coef: np.ndarray,
    global_intercept: np.ndarray,
    global_scaler_mean: np.ndarray,
    global_scaler_scale: np.ndarray,
    global_sample_count: int,
) -> Dict[str, np.ndarray]:
    """
    Perform Federated Averaging (FedAvg) weighted by local sample counts.

    Parameters
    ----------
    client_updates : list of dicts, each containing:
        - "coef"         (list / np.ndarray)
        - "intercept"    (float or list)
        - "sample_count" (int)
        - "client_id"    (str, for logging)
        Optional (used when present):
        - "scaler_mean"  (list / np.ndarray)
        - "scaler_scale" (list / np.ndarray)

    global_coef / global_intercept : current global model parameters
        (used to include the server's own "virtual client" contribution
        weighted by global_sample_count so existing knowledge is preserved).

    global_scaler_mean / global_scaler_scale : current scaler params
    global_sample_count : baseline weight for the existing global model

    Returns
    -------
    dict with keys: coef, intercept, scaler_mean, scaler_scale, total_samples
    """
    if not client_updates:
        return {
            "coef":         global_coef,
            "intercept":    global_intercept,
            "scaler_mean":  global_scaler_mean,
            "scaler_scale": global_scaler_scale,
            "total_samples": global_sample_count,
        }

    # Build a unified list including the existing global model as a "client"
    all_entries = [
        {
            "coef":         global_coef,
            "intercept":    global_intercept,
            "sample_count": global_sample_count,
            "client_id":    "__global__",
            "scaler_mean":  global_scaler_mean,
            "scaler_scale": global_scaler_scale,
        }
    ] + client_updates

    n_total = sum(int(e["sample_count"]) for e in all_entries)

    if n_total == 0:
        raise ValueError("FedAvg received zero total samples — cannot aggregate.")

    agg_coef      = np.zeros_like(global_coef, dtype=np.float64)
    agg_intercept = 0.0
    agg_s_mean    = np.zeros_like(global_scaler_mean,  dtype=np.float64)
    agg_s_scale   = np.zeros_like(global_scaler_scale, dtype=np.float64)

    for entry in all_entries:
        w = int(entry["sample_count"]) / n_total
        agg_coef      += w * np.asarray(entry["coef"],      dtype=np.float64)
        agg_intercept += w * float(np.asarray(entry["intercept"]).ravel()[0])

        # Scaler params are averaged only when present in the update
        if "scaler_mean" in entry and entry["scaler_mean"] is not None:
            agg_s_mean  += w * np.asarray(entry["scaler_mean"],  dtype=np.float64)
            agg_s_scale += w * np.asarray(entry["scaler_scale"], dtype=np.float64)
        else:
            # Fall back to global scaler weighted contribution
            agg_s_mean  += w * global_scaler_mean
            agg_s_scale += w * global_scaler_scale

    print(
        f"[aggregator] FedAvg over {len(client_updates)} client update(s), "
        f"total_samples={n_total}"
    )

    return {
        "coef":          agg_coef,
        "intercept":     np.array([agg_intercept]),
        "scaler_mean":   agg_s_mean,
        "scaler_scale":  agg_s_scale,
        "total_samples": n_total,
    }
