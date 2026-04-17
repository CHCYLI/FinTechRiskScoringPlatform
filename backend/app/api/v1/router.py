# backend/app/api/v1/router.py
"""
v1 router aggregator.

This file wires together all v1 endpoints so main.py can do:
app.include_router(api_router, prefix="/v1")
"""

from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.schema import router as schema_router
from app.api.v1.endpoints.validate import router as validate_router
from app.api.v1.endpoints.model import router as model_router

# NEW in Phase 2
from app.api.v1.endpoints.metrics import router as metrics_router

# NEW in Phase 3
from app.api.v1.endpoints.score import router as score_router
from app.api.v1.endpoints.explain import router as explain_router

api_router = APIRouter()

# Phase 1 endpoints
api_router.include_router(health_router)
api_router.include_router(schema_router)
api_router.include_router(validate_router)
api_router.include_router(model_router)

# Phase 2 endpoint
api_router.include_router(metrics_router)

# Phase 3 endpoint
api_router.include_router(score_router, tags=["scoring"])
api_router.include_router(explain_router, tags=["explain"])

