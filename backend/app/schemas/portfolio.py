from typing import List, Literal, Optional

from pydantic import BaseModel


GroupByField = Literal["region", "channel", "product"]


class PortfolioFilters(BaseModel):
    region: Optional[str] = None
    channel: Optional[str] = None
    product: Optional[str] = None


class PortfolioSummaryRow(BaseModel):
    group: str
    count: int
    avg_pd: float
    approve_count: int
    review_count: int
    reject_count: int
    approve_rate: float
    review_rate: float
    reject_rate: float
    bad_rate: Optional[float] = None


class PortfolioSummaryResponse(BaseModel):
    model_version: str
    group_by: GroupByField
    filters: PortfolioFilters
    rows: List[PortfolioSummaryRow]
