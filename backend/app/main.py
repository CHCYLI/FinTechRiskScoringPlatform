from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.router import router as v1_router

def create_app() -> FastAPI:
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.0.1",
    )

    app.include_router(v1_router)

    return app

app = create_app()
