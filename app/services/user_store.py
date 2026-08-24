from __future__ import annotations

import json
import logging

from app.core.config import settings
from app.services.features import UserFeatures, load_users_from_csv
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
        users = load_users_from_csv(path)
        logger.info("Loaded %d users into in-memory store", len(users))
        return cls(users)

    def get(self, user_id: int) -> UserFeatures | None:
        return self.users_by_id.get(user_id)


def load_split_users() -> tuple[dict[int, UserFeatures], dict[str, list[int]]]:
    """
    Loads the processed user table and the train/val/test user-id split
    produced by scripts/prepare_data.py. Shared by scripts/train.py and
    scripts/evaluate.py so both always parse the split the same way.
    """
    users = load_users_from_csv(settings.processed_dir / "users.csv")
    all_users = {u.user_id: u for u in users}
    split = json.loads((settings.processed_dir / "user_split.json").read_text())
    return all_users, split
