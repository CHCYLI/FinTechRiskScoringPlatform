from fastapi import APIRouter
from app.schemas.applicant import Applicant, ValidateResponse

router = APIRouter(tags=["validate"])


@router.post("/validate", response_model=ValidateResponse)
def validate_applicant(payload: Applicant):
    return ValidateResponse(ok=True, normalized=payload)
