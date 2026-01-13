import json
from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()

@router.get("/model/version")
def model_version():
    settings = get_settings()
    meta_path = settings.metadata_path

    payload = {
        "model_version": "untrained",
        "model_path": str(settings.model_path),
        "metadata_path": str(meta_path),
        "app_env": settings.app_env,
    }

    if meta_path.exists():
        try:
            payload["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
            payload["model_version"] = payload["metadata"].get("model_version", "unknown")
        except Exception as e:
            payload["metadata_error"] = str(e)

    return payload
