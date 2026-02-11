# backend/app/api/v1/endpoints/metrics.py
"""
GET /v1/metrics

Phase 2 endpoint: exposes model evaluation metrics stored in metadata.json.
If the model is not trained yet, return a friendly "untrained" response.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.artifact_loader import get_artifact_dir, load_metadata

router = APIRouter(tags=["model"])


@router.get("/metrics")
def get_metrics():
    meta = load_metadata()
    if meta is None:
        return {
            "status": "untrained",
            "artifact_dir": str(get_artifact_dir()),
            "detail": "No metadata.json found. Run Phase 2 training to generate artifacts.",
            "metrics": None,
        }

    # We store metrics as {"val": {...}, "test": {...}}
    return {
        "status": "ok",
        "version": meta.get("version"),
        "trained_at": meta.get("trained_at"),
        "metrics": meta.get("metrics", {}),
    }
