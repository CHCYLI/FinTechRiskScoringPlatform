from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[2] == backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """
    Full settings for later phases (2+),
    but safe to use starting from Phase 1.
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================
    # Basic
    # =========================
    app_name: str = Field(default="RiskScoringPlatform", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")

    # =========================
    # Environment / logging
    # =========================
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # =========================
    # API Server
    # =========================
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # =========================
    # Model artifacts
    # =========================
    model_dir: Path = Field(
        default=BACKEND_DIR / "app" / "ml" / "artifacts",
        alias="MODEL_DIR",
    )
    model_file: str = Field(default="model.joblib", alias="MODEL_FILE")
    metadata_file: str = Field(default="metadata.json", alias="METADATA_FILE")

    feature_schema_path: Path = Field(
        default=BACKEND_DIR / "app" / "ml" / "feature_schema.json",
        alias="FEATURE_SCHEMA_PATH",
    )

    # =========================
    # Decision thresholds
    # =========================
    threshold_approve: float = Field(default=0.30, alias="THRESHOLD_APPROVE")
    threshold_reject: float = Field(default=0.40, alias="THRESHOLD_REJECT")

    # =========================
    # Explainability
    # =========================
    shap_top_k: int = Field(default=5, alias="SHAP_TOP_K")
    shap_timeout_ms: int = Field(default=2000, alias="SHAP_TIMEOUT_MS")

    # =========================
    # Batch limits
    # =========================
    batch_max_rows: int = Field(default=5000, alias="BATCH_MAX_ROWS")

    # =========================
    # CORS
    # =========================
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    # =========================
    # Auth (optional)
    # =========================
    enable_auth: bool = Field(default=False, alias="ENABLE_AUTH")
    jwt_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")

    # =========================
    # Portfolio (optional)
    # =========================
    portfolio_data_path: Path = Field(
        default=PROJECT_ROOT / "ml" / "data" / "processed" / "train_real.csv",
        alias="PORTFOLIO_DATA_PATH",
    )

    @field_validator("model_dir", "feature_schema_path", "portfolio_data_path", mode="before")
    @classmethod
    def _resolve_relative_paths(cls, v: Any) -> Path:
        path = Path(v)
        if path.is_absolute():
            return path
        return (BACKEND_DIR / path).resolve()

    # -------- derived paths --------
    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_file

    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
