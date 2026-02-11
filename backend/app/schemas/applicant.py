from pydantic import BaseModel, Field, field_validator
from app.core.schema_loader import load_feature_schema


class Applicant(BaseModel):
    # --- segments: for portfolio split
    channel: str = Field(..., description="Acquisition channel")
    region: str = Field(..., description="Region bucket")
    product: str = Field(..., description="Product type")

    # --- features: for model to give score
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0, le=500000)
    employment_length: float = Field(..., ge=0, le=50)

    dti: float = Field(..., ge=0, le=3)
    utilization: float = Field(..., ge=0, le=1.5)

    delinquencies: int = Field(..., ge=0, le=20)
    history_length: float = Field(..., ge=0, le=40)

    tx_30d_count: int = Field(..., ge=0, le=5000)
    refund_rate_30d: float = Field(..., ge=0, le=1)
    active_days_30d: int = Field(..., ge=0, le=30)

    # --- validate enums using feature_schema.json
    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        allowed = set(load_feature_schema()["segments"]["channel"])
        if v not in allowed:
            raise ValueError(f"channel must be one of {sorted(allowed)}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        allowed = set(load_feature_schema()["segments"]["region"])
        if v not in allowed:
            raise ValueError(f"region must be one of {sorted(allowed)}")
        return v

    @field_validator("product")
    @classmethod
    def validate_product(cls, v: str) -> str:
        allowed = set(load_feature_schema()["segments"]["product"])
        if v not in allowed:
            raise ValueError(f"product must be one of {sorted(allowed)}")
        return v


class ValidateResponse(BaseModel):
    ok: bool
    normalized: Applicant
