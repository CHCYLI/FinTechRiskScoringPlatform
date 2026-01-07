from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/model/version")
def model_version():
    return {
        "model_version": "untrained",
        "model_path": settings.model_path,
        "metadata_path": settings.metadata_path,
    }
