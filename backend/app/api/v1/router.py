# backend/app/api/v1/router.py

from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.model import router as model_router
from app.api.v1.endpoints.schema import router as schema_router
from app.api.v1.endpoints.validate import router as validate_router

from app.api.v1.endpoints.metrics import router as metrics_router

router = APIRouter(prefix="/v1")

router.include_router(health_router, tags=["health"])
router.include_router(model_router, tags=["model"])
router.include_router(schema_router, tags=["schema"])
router.include_router(validate_router, tags=["validate"])

router.include_router(metrics_router, tags=["model"])
