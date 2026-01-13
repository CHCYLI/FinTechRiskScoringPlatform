from fastapi import APIRouter
from app.core.schema_loader import load_feature_schema

router = APIRouter(tags=["schema"])


@router.get("/schema")
def get_schema():
    return load_feature_schema()
