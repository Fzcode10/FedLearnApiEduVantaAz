"""
config.py
=========
Centralised configuration loader for the EduVanta Client Node.

Configuration priority (highest → lowest):
    1. Environment variables (EDUVANTA_*)
    2. Values from ``client_node/json/client_details.json``
    3. Hard-coded defaults

All filesystem paths are absolute and derived from the ``client_node/``
directory so the service works regardless of the current working directory.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, Optional

# ── directory anchors (absolute, never depend on cwd) ──────────────────────────
CLIENT_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(CLIENT_NODE_DIR, "json")
CLIENT_DETAILS_PATH = os.path.join(JSON_DIR, "client_details.json")


# ── config dataclass ───────────────────────────────────────────────────────────
@dataclass
class ClientConfig:
    """Immutable snapshot of resolved client configuration."""

    client_id: str
    server_url: str
    client_name: Optional[str]
    description: Optional[str]
    local_data_dir: str
    weights_dir: str
    epochs: int
    cache_path: str


# ── loader ─────────────────────────────────────────────────────────────────────
def _read_json_config() -> Dict:
    """Read ``client_details.json`` and return its contents as a dict.

    Returns an empty dict (with a warning) if the file is missing or
    contains invalid JSON — the caller will fall back to defaults.
    """
    if not os.path.isfile(CLIENT_DETAILS_PATH):
        print(
            f"[config] WARNING: {CLIENT_DETAILS_PATH} not found — "
            "using environment variables / defaults."
        )
        return {}

    try:
        with open(CLIENT_DETAILS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Root element is not a JSON object")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            f"[config] WARNING: {CLIENT_DETAILS_PATH} is malformed ({exc}) — "
            "using environment variables / defaults."
        )
        return {}


def load_client_config() -> ClientConfig:
    """Build a :class:`ClientConfig` by merging env vars, JSON, and defaults.

    Priority: environment variable  ▸  JSON value  ▸  hard-coded default.
    """
    json_cfg = _read_json_config()

    # --- individual fields ---------------------------------------------------
    default_hostname = f"client-{socket.gethostname()}"

    client_id = os.getenv(
        "EDUVANTA_CLIENT_ID",
        json_cfg.get("client_id", default_hostname),
    )

    server_url = os.getenv(
        "EDUVANTA_SERVER_URL",
        json_cfg.get("server_url", "http://localhost:8000"),
    )

    client_name: Optional[str] = json_cfg.get("client_name")
    description: Optional[str] = json_cfg.get("description")

    local_data_dir = os.getenv(
        "EDUVANTA_LOCAL_DATA",
        os.path.join(CLIENT_NODE_DIR, "local_data"),
    )

    weights_dir = os.getenv(
        "EDUVANTA_WEIGHTS_DIR",
        os.path.join(CLIENT_NODE_DIR, "weights"),
    )

    epochs_str = os.getenv("EDUVANTA_EPOCHS")
    if epochs_str is not None:
        try:
            epochs = int(epochs_str)
        except ValueError:
            print(
                f"[config] WARNING: EDUVANTA_EPOCHS='{epochs_str}' is not an "
                "integer — falling back to default (3)."
            )
            epochs = 3
    else:
        epochs = 3

    cache_path = os.path.join(weights_dir, "local_weights_cache.npz")

    # --- ensure data directories exist ---------------------------------------
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    cfg = ClientConfig(
        client_id=client_id,
        server_url=server_url,
        client_name=client_name,
        description=description,
        local_data_dir=local_data_dir,
        weights_dir=weights_dir,
        epochs=epochs,
        cache_path=cache_path,
    )

    # --- print summary -------------------------------------------------------
    print("[config] -- EduVanta Client Configuration --")
    print(f"[config]   client_id      = {cfg.client_id}")
    print(f"[config]   server_url     = {cfg.server_url}")
    if cfg.client_name:
        print(f"[config]   client_name    = {cfg.client_name}")
    if cfg.description:
        print(f"[config]   description    = {cfg.description}")
    print(f"[config]   local_data_dir = {cfg.local_data_dir}")
    print(f"[config]   weights_dir    = {cfg.weights_dir}")
    print(f"[config]   epochs         = {cfg.epochs}")
    print(f"[config]   cache_path     = {cfg.cache_path}")
    print("[config] ----------------------------------------")

    return cfg
