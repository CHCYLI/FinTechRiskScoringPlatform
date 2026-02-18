import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import setup_logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up...")
    logger.info("env=%s schema=%s", settings.app_env, settings.feature_schema_path)
    scheduler = AsyncIOScheduler()
    scheduler.start()
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        logger.info("API shutting down...")


# don't touch this
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router)
