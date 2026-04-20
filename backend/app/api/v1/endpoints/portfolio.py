from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.portfolio import PortfolioSummaryResponse
from app.services.portfolio import build_portfolio_summary

router = APIRouter()


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse)
def get_portfolio_summary(
    group_by: Literal["region", "channel", "product"] = Query(...),
    region: Optional[str] = Query(default=None),
    channel: Optional[str] = Query(default=None),
    product: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    try:
        return build_portfolio_summary(
            group_by=group_by,
            region=region,
            channel=channel,
            product=product,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio summary failed: {e}")
