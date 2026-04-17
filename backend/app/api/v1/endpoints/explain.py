from fastapi import APIRouter, HTTPException

from app.schemas.applicant import Applicant
from app.schemas.explain import ExplainResponse
from app.services.explainability import explain_one_placeholder

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
def explain(applicant: Applicant):
    try:
        return explain_one_placeholder(applicant.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")
