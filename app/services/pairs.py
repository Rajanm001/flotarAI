from __future__ import annotations

import numpy as np

from app.services.features import UserFeatures, user_feature_vector
from app.services.labeling import RARITY_VECTOR, is_relevant_batch
from app.services.retrieval import CandidateRetriever


def build_training_pairs(
    target_users: list[UserFeatures],
    retriever: CandidateRetriever,
    max_targets: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each target user, retrieve their Stage A candidate pool and build one
    labeled feature row per (target, candidate) pair, batched with numpy
    rather than per-pair Python loops. Training/eval pairs are restricted to
    the retrieved pool -- the same distribution the ranker sees at inference
    time -- rather than random pairs from the whole dataset.

    max_targets subsamples target users (a 100-candidate pool per target
    already gives ~100x pairs; a few thousand targets is enough signal for an
    MLP this small, and keeps pair-generation time reasonable).
    """
    rng = np.random.default_rng(seed)
    if max_targets is not None and len(target_users) > max_targets:
        target_users = [target_users[i] for i in rng.choice(len(target_users), max_targets, replace=False)]

    candidate_matrix = retriever._interest_matrix
    candidate_ages = retriever._ages
    candidate_cities = retriever._cities
    candidate_countries = retriever._countries
    candidate_genders = retriever._genders
    candidate_user_vecs = np.concatenate(
        [(candidate_ages / 100.0)[:, None], candidate_matrix], axis=1
    ).astype(np.float32)

    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for target in target_users:
        idx = retriever.retrieve_indices(target)

        cand_interest = candidate_matrix[idx].astype(bool)
        target_interest = target.interest_vector.astype(bool)

        intersection = np.logical_and(cand_interest, target_interest).sum(axis=1)
        union = np.logical_or(cand_interest, target_interest).sum(axis=1)
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union != 0)
        shared_count = intersection.astype(np.float32)
        same_city = (candidate_cities[idx] == target.city).astype(np.float32)
        same_country = (candidate_countries[idx] == target.country).astype(np.float32)
        same_gender = (candidate_genders[idx] == target.gender).astype(np.float32)
        age_diff = np.abs(candidate_ages[idx] - target.age).astype(np.float32)

        pairwise = np.stack([jaccard, shared_count, same_city, same_country, same_gender, age_diff], axis=1)

        target_vec = user_feature_vector(target)
        target_block = np.tile(target_vec, (len(idx), 1))
        candidate_block = candidate_user_vecs[idx]

        batch_rows = np.concatenate([target_block, candidate_block, pairwise], axis=1)
        rows.append(batch_rows)

        shared_mask = cand_interest & target_interest
        rare_bonus = (shared_mask.astype(np.float32) * RARITY_VECTOR).sum(axis=1)
        batch_labels = is_relevant_batch(
            jaccard=jaccard,
            same_country=same_country,
            same_city=same_city,
            age_diff=age_diff,
            rare_bonus=rare_bonus,
            rng=rng,
        ).astype(np.float32)
        labels.append(batch_labels)

    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0)
