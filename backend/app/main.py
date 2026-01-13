import logging

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger("app")

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

app.include_router(v1_router)


@app.on_event("startup")
def on_startup():
    logger.info("API starting up...")
    logger.info("env=%s schema=%s", settings.app_env, settings.feature_schema_path)
