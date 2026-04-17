from typing import List

from pydantic import BaseModel


class ExplainResponse(BaseModel):
    model_version: str
    top_features: List[str]
    reasons: List[str]
