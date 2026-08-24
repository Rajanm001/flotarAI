import numpy as np

from app.services.features import UserFeatures, interest_multi_hot
from app.services.retrieval import CandidateRetriever


def make_user(user_id: int, interests: list[str], city: str, country: str, age: float, gender: str = "Male") -> UserFeatures:
    return UserFeatures(
        user_id=user_id,
        name=f"User {user_id}",
        gender=gender,
        age=age,
        city=city,
        country=country,
        interests=interests,
        interest_vector=interest_multi_hot(interests),
    )


def test_retrieve_excludes_target_user():
    users = [
        make_user(1, ["Music"], "Delhi", "India", 25.0),
        make_user(2, ["Music"], "Delhi", "India", 25.0),
        make_user(3, ["Sports"], "Paris", "France", 40.0),
    ]
    retriever = CandidateRetriever(users, pool_size=2)
    results = retriever.retrieve(users[0])
    assert users[0].user_id not in [u.user_id for u in results]


def test_retrieve_prefers_similar_interests_and_location():
    target = make_user(1, ["Technology", "Music"], "Delhi", "India", 25.0)
    close_match = make_user(2, ["Technology", "Music"], "Delhi", "India", 26.0)
    far_match = make_user(3, ["Gardening"], "Lima", "Peru", 70.0)

    users = [target, close_match, far_match]
    retriever = CandidateRetriever(users, pool_size=2)
    results = retriever.retrieve(target)

    assert results[0].user_id == close_match.user_id


def test_retrieve_pool_size_respected():
    users = [make_user(i, ["Music"], "Delhi", "India", 25.0) for i in range(50)]
    retriever = CandidateRetriever(users, pool_size=10)
    results = retriever.retrieve(users[0])
    assert len(results) == 10


def test_retrieve_handles_pool_smaller_than_requested_size():
    users = [make_user(i, ["Music"], "Delhi", "India", 25.0) for i in range(5)]
    retriever = CandidateRetriever(users, pool_size=100)
    results = retriever.retrieve(users[0])
    assert len(results) == 4
