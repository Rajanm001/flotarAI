"""
Loads the raw CSV, builds per-user feature vectors, and writes an 80/10/10
train/val/test split of *target user ids* to data/processed/.

The split is over target users (not rows of a labeled interaction table)
because the dataset has no ground-truth follow graph -- see README for the
implicit-relevance-signal trade-off this implies for training. Splitting by
user id keeps a user's candidate pool and generated pairs entirely within one
split, so no target user's data leaks across train/val/test.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logging import configure_logging
from app.services.features import build_user_features

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset from %s", settings.raw_dataset_path)
    df = pd.read_csv(settings.raw_dataset_path)
    logger.info("Loaded %d users", len(df))

    users = [build_user_features(row) for _, row in df.iterrows()]
    user_ids = np.array([u.user_id for u in users])

    rng = np.random.default_rng(settings.random_seed)
    shuffled = rng.permutation(user_ids)

    n = len(shuffled)
    n_train = int(n * settings.train_split)
    n_val = int(n * settings.val_split)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(train_ids), len(val_ids), len(test_ids)
    )

    split = {
        "train": train_ids.tolist(),
        "val": val_ids.tolist(),
        "test": test_ids.tolist(),
    }
    split_path = settings.processed_dir / "user_split.json"
    split_path.write_text(json.dumps(split))
    logger.info("Wrote split to %s", split_path)

    df.to_csv(settings.processed_dir / "users.csv", index=False)
    logger.info("Wrote cleaned user table to %s", settings.processed_dir / "users.csv")


if __name__ == "__main__":
    main()
