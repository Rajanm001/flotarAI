from __future__ import annotations

import ast
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.config import settings

# Fixed vocabulary observed in the provided dataset (29 categories). Kept as a
# constant rather than derived at request-time so a single test user's
# interests always map to the same feature indices the model was trained on.
INTEREST_VOCAB = [
    "Art", "Beauty", "Books", "Business and entrepreneurship",
    "Cars and automobiles", "Cooking", "DIY and crafts",
    "Education and learning", "Fashion", "Finance and investments",
    "Fitness", "Food and dining", "Gaming", "Gardening",
    "Health and wellness", "History", "Movies", "Music", "Nature",
    "Outdoor activities", "Parenting and family", "Pets", "Photography",
    "Politics", "Science", "Social causes and activism", "Sports",
    "Technology", "Travel",
]
INTEREST_INDEX = {name: i for i, name in enumerate(INTEREST_VOCAB)}


def parse_interests(raw: str) -> list[str]:
    """Parse the CSV's stringified-list Interests column, e.g. "'Gaming', 'Music'"."""
    if not raw or not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(f"[{raw}]") if not raw.strip().startswith("[") else ast.literal_eval(raw)
        return [str(item).strip() for item in parsed]
    except (ValueError, SyntaxError):
        return [token.strip().strip("'\"") for token in raw.split(",") if token.strip()]


def interest_multi_hot(interests: list[str]) -> np.ndarray:
    vec = np.zeros(len(INTEREST_VOCAB), dtype=np.float32)
    for name in interests:
        idx = INTEREST_INDEX.get(name)
        if idx is not None:
            vec[idx] = 1.0
    return vec


class MalformedDobError(ValueError):
    """Raised when a DOB string cannot be parsed as an ISO date."""


def age_from_dob(dob: str, reference_date: dt.date | None = None) -> float:
    """
    Raises MalformedDobError rather than silently defaulting to age 0.0 --
    a wrong-but-plausible age (0.0) would silently corrupt every age-based
    feature and similarity score for that user with no signal that anything
    went wrong, which is worse than failing loudly at the boundary (API
    validation, or a clear error during offline data preparation).

    reference_date defaults to settings.age_reference_date, read at call
    time rather than bound as a module-level constant at import time, so a
    runtime override of the setting (e.g. via FLOTER_AGE_REFERENCE_DATE)
    takes effect without needing a process restart mid-import.
    """
    if reference_date is None:
        reference_date = settings.age_reference_date
    try:
        birth = dt.date.fromisoformat(dob)
    except (ValueError, TypeError) as exc:
        raise MalformedDobError(f"Could not parse DOB '{dob}' as an ISO date (YYYY-MM-DD)") from exc
    years = reference_date.year - birth.year - (
        (reference_date.month, reference_date.day) < (birth.month, birth.day)
    )
    return float(years)


@dataclass(frozen=True)
class UserFeatures:
    user_id: int
    name: str
    gender: str
    age: float
    city: str
    country: str
    interests: list[str]
    interest_vector: np.ndarray

    @classmethod
    def from_profile(cls, user_id: int, name: str, gender: str, dob: str, interests: list[str], city: str, country: str) -> "UserFeatures":
        """Builds a UserFeatures from an API-layer profile (already-validated dob/interests)."""
        return cls(
            user_id=user_id,
            name=name,
            gender=gender,
            age=age_from_dob(dob),
            city=city,
            country=country,
            interests=interests,
            interest_vector=interest_multi_hot(interests),
        )


def build_user_features(row: pd.Series) -> UserFeatures:
    try:
        user_id = int(row["UserID"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Row has a non-integer UserID: {row.get('UserID')!r}") from exc

    try:
        age = age_from_dob(row["DOB"])
    except MalformedDobError as exc:
        raise MalformedDobError(f"user_id={user_id}: {exc}") from exc

    interests = parse_interests(row["Interests"])
    return UserFeatures(
        user_id=user_id,
        name=str(row["Name"]),
        gender=str(row["Gender"]),
        age=age,
        city=str(row["City"]),
        country=str(row["Country"]),
        interests=interests,
        interest_vector=interest_multi_hot(interests),
    )


def load_users_from_csv(path) -> list[UserFeatures]:
    """
    Single source of truth for turning a raw or processed users CSV into
    UserFeatures, used identically by the live API's UserStore, the
    training pipeline, and offline evaluation -- so all three always see
    the same population built the same way, and duplicate UserIDs are
    caught here rather than silently resolving to "last row wins" in
    whichever data structure happens to key on user_id downstream.
    """
    df = pd.read_csv(path)
    duplicated = df["UserID"][df["UserID"].duplicated()].unique()
    if len(duplicated) > 0:
        raise ValueError(
            f"Found {len(duplicated)} duplicate UserID value(s) in {path}: {sorted(duplicated)[:10]}"
            f"{'...' if len(duplicated) > 10 else ''}. Refusing to load an ambiguous user pool."
        )
    return [build_user_features(row) for _, row in df.iterrows()]


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return float(intersection / union)


def pairwise_features(target: UserFeatures, candidate: UserFeatures) -> np.ndarray:
    """
    Hand-engineered pairwise features between a target user and one candidate.
    Location is derived from City/Country (the provided CSV has no lat/long),
    documented as a deliberate trade-off in the README.
    """
    shared_interests = float(np.logical_and(target.interest_vector, candidate.interest_vector).sum())
    jaccard = jaccard_similarity(target.interest_vector, candidate.interest_vector)
    same_city = 1.0 if target.city == candidate.city else 0.0
    same_country = 1.0 if target.country == candidate.country else 0.0
    same_gender = 1.0 if target.gender == candidate.gender else 0.0
    age_diff = abs(target.age - candidate.age)

    return np.array(
        [jaccard, shared_interests, same_city, same_country, same_gender, age_diff],
        dtype=np.float32,
    )


PAIRWISE_FEATURE_NAMES = [
    "interest_jaccard",
    "shared_interest_count",
    "same_city",
    "same_country",
    "same_gender",
    "age_diff",
]


def user_feature_vector(user: UserFeatures) -> np.ndarray:
    """Standalone per-user feature vector: normalized age + interest multi-hot."""
    normalized_age = np.array([user.age / 100.0], dtype=np.float32)
    return np.concatenate([normalized_age, user.interest_vector])


def batch_user_feature_matrix(users: list[UserFeatures]) -> np.ndarray:
    """Vectorized equivalent of stacking user_feature_vector() over many users."""
    ages = np.array([u.age for u in users], dtype=np.float32) / 100.0
    interest_matrix = np.stack([u.interest_vector for u in users]).astype(np.float32)
    return np.concatenate([ages[:, None], interest_matrix], axis=1)


def batch_pairwise_features(target: UserFeatures, candidates: list[UserFeatures]) -> np.ndarray:
    """
    Vectorized equivalent of stacking pairwise_features(target, c) over many
    candidates -- the single source of truth for the six pairwise features,
    used identically by offline training-pair generation and online serving
    so the two can never silently drift out of sync with each other.
    """
    target_interest = target.interest_vector.astype(bool)
    candidate_interest = np.stack([c.interest_vector for c in candidates]).astype(bool)

    intersection = np.logical_and(candidate_interest, target_interest).sum(axis=1)
    union = np.logical_or(candidate_interest, target_interest).sum(axis=1)
    jaccard = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union != 0
    )
    shared_interests = intersection.astype(np.float32)
    same_city = np.array([1.0 if c.city == target.city else 0.0 for c in candidates], dtype=np.float32)
    same_country = np.array([1.0 if c.country == target.country else 0.0 for c in candidates], dtype=np.float32)
    same_gender = np.array([1.0 if c.gender == target.gender else 0.0 for c in candidates], dtype=np.float32)
    age_diff = np.array([abs(c.age - target.age) for c in candidates], dtype=np.float32)

    return np.stack([jaccard, shared_interests, same_city, same_country, same_gender, age_diff], axis=1)


USER_FEATURE_DIM = 1 + len(INTEREST_VOCAB)
PAIRWISE_FEATURE_DIM = len(PAIRWISE_FEATURE_NAMES)
RANKER_INPUT_DIM = USER_FEATURE_DIM * 2 + PAIRWISE_FEATURE_DIM
