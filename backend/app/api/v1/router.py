from fastapi import APIRouter
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.model import router as model_router

router = APIRouter(prefix="/v1")
router.include_router(health_router, tags=["health"])
router.include_router(model_router, tags=["model"])
