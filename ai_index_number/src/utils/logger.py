# src/utils/logger.py
# Author: [Your Name] | Index: [Your Index Number]
# Handles structured JSON logging for each RAG pipeline query

import json
import os
from datetime import datetime
from typing import Any, Dict

LOG_PATH = os.path.join(os.path.dirname(__file__), "../../outputs/logs.json")


def _load_logs() -> list:
    """Load existing logs from file."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def log_query(entry: Dict[str, Any]) -> None:
    """Append a structured log entry to logs.json."""
    logs = _load_logs()
    entry["timestamp"] = datetime.utcnow().isoformat()
    logs.append(entry)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2, default=str)


def get_logs() -> list:
    """Return all stored logs."""
    return _load_logs()
