from fastapi import APIRouter
from app.core.config import settings
router = APIRouter()

@router.get("/model/version")
def model_version():
    # Phase 0: placeholder. After phase 2 read metadata.json
    return {
        "model_version": "untrained",
        "metadata_path": settings.metadata_path,
        "note": "Phase 0 placeholder. Train model in Phase 2 to populate metadata.json."
    }
