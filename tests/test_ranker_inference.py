import numpy as np
import torch

from app.models.ranker import PairwiseRanker
from app.services.features import RANKER_INPUT_DIM, UserFeatures, interest_multi_hot
from app.services.ranker_inference import rank_top_k, score_candidates


def make_user(user_id: int, interests: list[str], city: str = "Delhi", country: str = "India", age: float = 25.0) -> UserFeatures:
    return UserFeatures(
        user_id=user_id,
        name=f"User {user_id}",
        gender="Male",
        age=age,
        city=city,
        country=country,
        interests=interests,
        interest_vector=interest_multi_hot(interests),
    )


def test_score_candidates_empty_list_returns_empty_array():
    model = PairwiseRanker(input_dim=RANKER_INPUT_DIM)
    target = make_user(1, ["Music"])
    scores = score_candidates(model, target, [])
    assert scores.shape == (0,)


def test_rank_top_k_empty_candidates_returns_empty_list():
    model = PairwiseRanker(input_dim=RANKER_INPUT_DIM)
    target = make_user(1, ["Music"])
    result = rank_top_k(model, target, [], k=10)
    assert result == []


def test_score_candidates_matches_manual_feature_construction():
    """Vectorized batch scoring must produce identical scores to scoring one
    candidate at a time -- guards the score_candidates vectorization against
    silently diverging from a naive per-candidate computation."""
    model = PairwiseRanker(input_dim=RANKER_INPUT_DIM)
    model.eval()
    target = make_user(1, ["Music", "Technology"])
    candidates = [
        make_user(2, ["Music"], city="Delhi", country="India", age=26.0),
        make_user(3, ["Gardening"], city="Lima", country="Peru", age=70.0),
    ]

    batch_scores = score_candidates(model, target, candidates)
    single_scores = [score_candidates(model, target, [c])[0] for c in candidates]

    np.testing.assert_allclose(batch_scores, single_scores, rtol=1e-5)
