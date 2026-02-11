# backend/app/core/artifact_loader.py
"""
Centralized artifact loader (Phase 2+).

Why:
- Endpoints should NOT hardcode paths or repeat IO logic.
- We want one place that knows:
  - where artifacts live (default: backend/app/ml/artifacts)
  - how to load metadata.json (cached)
  - how to load model.joblib (Phase 3+)

Design notes:
- In dev, you typically restart uvicorn when artifacts change, so simple caching is fine.
- If you later want hot-reload, you can add mtime-based invalidation.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


def default_artifact_dir() -> Path:
    """
    Default artifact directory = <backend>/app/ml/artifacts
    This file is at: backend/app/core/artifact_loader.py
    - parents[0] -> core
    - parents[1] -> app
    """
    app_dir = Path(__file__).resolve().parents[1]
    return app_dir / "ml" / "artifacts"


def get_artifact_dir() -> Path:
    """
    Decide artifact directory in a production-friendly way:
    1) env var MODEL_DIR (or ARTIFACT_DIR) if set
    2) fallback to default within repo
    """
    env_dir = os.getenv("MODEL_DIR") or os.getenv("ARTIFACT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return default_artifact_dir()


def metadata_path() -> Path:
    return get_artifact_dir() / "metadata.json"


def model_path() -> Path:
    return get_artifact_dir() / "model.joblib"


@lru_cache(maxsize=1)
def load_metadata() -> Optional[Dict[str, Any]]:
    """
    Load metadata.json (cached).
    Returns None if not found, which endpoints can interpret as "untrained".
    """
    path = metadata_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        # If metadata is corrupted, surface it clearly
        raise RuntimeError(f"Failed to read metadata.json at {path}: {e}") from e


@lru_cache(maxsize=1)
def load_model() -> Any:
    """
    Load model.joblib (cached). Not required for Phase 2 endpoints,
    but Phase 3 (/score) will use it.
    """
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run ml/train.py first.")
    return joblib.load(path)


def clear_artifact_cache() -> None:
    """Call this if you want to refresh artifacts without restarting (optional)."""
    load_metadata.cache_clear()
    load_model.cache_clear()
