from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfileRequest(BaseModel):
    user_id: int = Field(..., description="Unique id for the target user")
    name: str
    gender: str
    dob: str = Field(..., description="ISO date, YYYY-MM-DD")
    interests: list[str]
    city: str
    country: str


class RecommendationItem(BaseModel):
    user_id: int
    name: str
    city: str
    country: str
    score: float


class RecommendationResponse(BaseModel):
    target_user_id: int
    candidate_pool_size: int
    recommendations: list[RecommendationItem]
