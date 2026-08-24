from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator

from app.services.features import INTEREST_VOCAB

_INTEREST_SET = set(INTEREST_VOCAB)


class UserProfileRequest(BaseModel):
    user_id: int = Field(..., description="Unique id for the target user")
    name: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1)
    dob: str = Field(..., description="ISO date, YYYY-MM-DD")
    interests: list[str] = Field(
        ..., min_length=1, description=f"Must be drawn from: {', '.join(INTEREST_VOCAB)}"
    )
    city: str = Field(..., min_length=1)
    country: str = Field(..., min_length=1)

    @field_validator("dob")
    @classmethod
    def dob_must_be_iso_date(cls, value: str) -> str:
        try:
            dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"dob must be an ISO date (YYYY-MM-DD), got {value!r}") from exc
        return value

    @field_validator("interests")
    @classmethod
    def interests_must_be_known_and_non_blank(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("interests must not contain blank entries")
        unknown = sorted(set(cleaned) - _INTEREST_SET)
        if unknown:
            raise ValueError(
                f"unrecognized interests {unknown}; must be drawn from the fixed vocabulary: {INTEREST_VOCAB}"
            )
        return cleaned


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
