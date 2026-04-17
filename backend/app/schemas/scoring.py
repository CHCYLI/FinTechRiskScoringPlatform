# backend/app/schemas/scoring.py
from typing import Literal, List
from pydantic import BaseModel


Decision = Literal["Approve", "Review", "Reject"]


class ScoreResponse(BaseModel):
    pd: float
    decision: Decision
    model_version: str
    thresholds: dict


class BatchScoreItem(BaseModel):
    index: int
    pd: float
    decision: Decision


class BatchScoreResponse(BaseModel):
    model_version: str
    thresholds: dict
    results: List[BatchScoreItem]
