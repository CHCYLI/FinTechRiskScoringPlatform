from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    app_name: str = Field(default="RiskScoringPlatform", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")

    
    
    # ✅ 关键：extra="ignore" -> .env 里多余键不会导致崩溃
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================
    # Environment / logging
    # =========================
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # =========================
    # API Server (match your .env)
    # =========================
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # =========================
    # Model artifacts (match your .env)
    # =========================
    model_dir: str = Field(default="app/ml/artifacts", alias="MODEL_DIR")
    model_file: str = Field(default="model.joblib", alias="MODEL_FILE")
    metadata_file: str = Field(default="metadata.json", alias="METADATA_FILE")

    feature_schema_path: str = Field(
        default="app/ml/feature_schema.json",
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
    # Auth (optional, match your .env)
    # =========================
    enable_auth: bool = Field(default=False, alias="ENABLE_AUTH")
    jwt_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")

    # =========================
    # Portfolio (optional)
    # =========================
    portfolio_data_path: str = Field(
        default="app/ml/data/portfolio_sample.csv",
        alias="PORTFOLIO_DATA_PATH",
    )

    # -------- derived paths --------
    @property
    def model_path(self) -> str:
        return str(Path(self.model_dir) / self.model_file)

    @property
    def metadata_path(self) -> str:
        return str(Path(self.model_dir) / self.metadata_file)

    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


settings = Settings()
