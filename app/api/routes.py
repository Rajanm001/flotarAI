from __future__ import annotations

from fastapi import APIRouter, Request

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
    return get_recommendations(profile, store, model)
