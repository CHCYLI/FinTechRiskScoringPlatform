# backend/app/schemas/applicant.py
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from app.core.schema_loader import load_feature_schema


class Applicant(BaseModel):
    # --- segments: for portfolio split
    channel: Optional[str] = Field(default=None, description="Acquisition channel")
    region: Optional[str] = Field(default=None, description="Region bucket")
    product: Optional[str] = Field(default=None, description="Product type")

    # --- features: for model to give score
    age: Optional[int] = Field(default=None, ge=18, le=100)
    income: Optional[float] = Field(default=None, ge=0, le=500000)
    employment_length: Optional[float] = Field(default=None, ge=0, le=50)

    dti: Optional[float] = Field(default=None, ge=0, le=3)
    utilization: Optional[float] = Field(default=None, ge=0, le=1.5)

    delinquencies: Optional[int] = Field(default=None, ge=0, le=20)
    history_length: Optional[float] = Field(default=None, ge=0, le=40)

    tx_30d_count: Optional[int] = Field(default=None, ge=0, le=5000)
    refund_rate_30d: Optional[float] = Field(default=None, ge=0, le=1)
    active_days_30d: Optional[int] = Field(default=None, ge=0, le=30)

    # --- validate enums using feature_schema.json (only if not None)
    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = set(load_feature_schema()["segments"]["channel"])
            if v not in allowed:
                raise ValueError(f"channel must be one of {sorted(allowed)}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = set(load_feature_schema()["segments"]["region"])
            if v not in allowed:
                raise ValueError(f"region must be one of {sorted(allowed)}")
        return v

    @field_validator("product")
    @classmethod
    def validate_product(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = set(load_feature_schema()["segments"]["product"])
            if v not in allowed:
                raise ValueError(f"product must be one of {sorted(allowed)}")
        return v


class ValidateResponse(BaseModel):
    ok: bool
    normalized: Applicant
