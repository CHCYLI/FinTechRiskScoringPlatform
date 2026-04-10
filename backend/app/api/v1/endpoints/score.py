# backend/app/api/v1/endpoints/score.py
from fastapi import APIRouter, HTTPException
from app.schemas.applicant import Applicant
from app.schemas.scoring import ScoreResponse, BatchScoreResponse
from app.services.inference import score_one, score_batch

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    try:
        return score_one(applicant.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch_json(applicants: list[Applicant]):
    try:
        payload = [a.model_dump() for a in applicants]
        return score_batch(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {e}")