from __future__ import annotations

from app.core.config import settings
from app.models.ranker import PairwiseRanker
from app.schemas.recommendation import (
    RecommendationItem,
    RecommendationResponse,
    UserProfileRequest,
)
from app.services.features import UserFeatures, age_from_dob, interest_multi_hot
from app.services.ranker_inference import rank_top_k
from app.services.user_store import UserStore


def profile_to_user_features(profile: UserProfileRequest) -> UserFeatures:
    return UserFeatures(
        user_id=profile.user_id,
        name=profile.name,
        gender=profile.gender,
        age=age_from_dob(profile.dob),
        city=profile.city,
        country=profile.country,
        interests=profile.interests,
        interest_vector=interest_multi_hot(profile.interests),
    )


def get_recommendations(
    profile: UserProfileRequest,
    store: UserStore,
    model: PairwiseRanker,
    top_k: int = settings.final_recommendation_count,
) -> RecommendationResponse:
    target = profile_to_user_features(profile)
    candidates = store.retriever.retrieve(target)
    ranked = rank_top_k(model, target, candidates, k=top_k)

    return RecommendationResponse(
        target_user_id=target.user_id,
        candidate_pool_size=len(candidates),
        recommendations=[
            RecommendationItem(
                user_id=candidate.user_id,
                name=candidate.name,
                city=candidate.city,
                country=candidate.country,
                score=score,
            )
            for candidate, score in ranked
        ],
    )
