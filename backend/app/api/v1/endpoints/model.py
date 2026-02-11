# backend/app/api/v1/endpoints/model.py
"""
GET /v1/model/version

Phase 2 behavior:
- If metadata exists: return version info from metadata.json
- If not trained: return "untrained" + artifact_dir
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.artifact_loader import get_artifact_dir, load_metadata

router = APIRouter(tags=["model"])


@router.get("/model/version")
def get_model_version():
    meta = load_metadata()
    if meta is None:
        return {
            "status": "untrained",
            "artifact_dir": str(get_artifact_dir()),
            "detail": "No artifacts found. Run ml/train.py to generate model.joblib and metadata.json.",
        }

    return {
        "status": "ok",
        "model_name": meta.get("model_name"),
        "version": meta.get("version"),
        "trained_at": meta.get("trained_at"),
        "feature_schema_sha256": meta.get("feature_schema_sha256"),
        "features": meta.get("features", []),
        "segments": meta.get("segments", []),
        "thresholds": meta.get("thresholds", {}),
        "artifact_dir": str(get_artifact_dir()),
        "artifact_files": meta.get("artifact_files", {}),
    }
