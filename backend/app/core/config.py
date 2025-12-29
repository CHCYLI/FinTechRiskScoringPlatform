from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = Field(default="RiskScoringPlatform", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    threshold_approve: float = Field(default=0.30, alias="THRESHOLD_APPROVE")
    threshold_reject: float = Field(default=0.40, alias="THRESHOLD_REJECT")

    model_path: str = Field(default="app/ml/artifacts/model.joblib", alias="MODEL_PATH")
    metadata_path: str = Field(default="app/ml/artifacts/metadata.json", alias="METADATA_PATH")

    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_alg: str = Field(default="HS256", alias="JWT_ALG")

    def cors_origin_list(self) -> List[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


settings = Settings()
