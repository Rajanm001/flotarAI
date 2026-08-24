from __future__ import annotations

import logging

import pandas as pd

from app.core.config import settings
from app.services.features import UserFeatures, build_user_features
from app.services.retrieval import CandidateRetriever

logger = logging.getLogger(__name__)


class UserStore:
    """Holds the in-memory user pool and retriever, built once at startup."""

    def __init__(self, users: list[UserFeatures]):
        self.users_by_id: dict[int, UserFeatures] = {u.user_id: u for u in users}
        self.retriever = CandidateRetriever(users, pool_size=settings.candidate_pool_size)

    @classmethod
    def from_csv(cls, path=None) -> "UserStore":
        path = path or settings.raw_dataset_path
        logger.info("Loading user pool from %s", path)
        df = pd.read_csv(path)
        users = [build_user_features(row) for _, row in df.iterrows()]
        logger.info("Loaded %d users into in-memory store", len(users))
        return cls(users)

    def get(self, user_id: int) -> UserFeatures | None:
        return self.users_by_id.get(user_id)
