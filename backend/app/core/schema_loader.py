import json
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings


@lru_cache
def load_feature_schema() -> dict:
    settings = get_settings()
    path: Path = settings.feature_schema_path

    if not path.exists():
        raise FileNotFoundError(f"FEATURE_SCHEMA_PATH not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 简单 sanity check（防止 schema 文件缺关键字段）
    if "features" not in data or "segments" not in data:
        raise ValueError("feature_schema.json must contain 'features' and 'segments' keys")

    return data
