from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.recommendation import RecommendationResponse, UserProfileRequest
from app.services.recommender import get_recommendations

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/recommendations", response_model=RecommendationResponse)
def recommendations(profile: UserProfileRequest, request: Request) -> RecommendationResponse:
    store = request.app.state.user_store
    model = request.app.state.ranker_model

    if not profile.interests:
        raise HTTPException(status_code=422, detail="interests must be a non-empty list")

    return get_recommendations(profile, store, model)
