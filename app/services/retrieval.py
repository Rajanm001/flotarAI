from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.features import UserFeatures, jaccard_similarity


@dataclass(frozen=True)
class PopulationArrays:
    """
    Public, read-only view of the retriever's precomputed per-user arrays.

    Exposed as a stable accessor (rather than callers reaching into
    CandidateRetriever's underscore-prefixed attributes directly) so
    downstream consumers -- training pair generation, offline evaluation --
    depend on an explicit contract instead of the retriever's private
    representation, which is free to change without breaking them.
    """

    interest_matrix: np.ndarray
    ages: np.ndarray
    cities: np.ndarray
    countries: np.ndarray
    genders: np.ndarray


class CandidateRetriever:
    """
    Stage A: fast, in-memory candidate generation.

    Scores every other user against the target with a cheap weighted sum of
    interest-Jaccard similarity, same-country/same-city bonuses, and an age-gap
    penalty, fully vectorized over the user pool rather than looping in Python.
    Chosen over an inverted interest index because the pool here is small
    enough (tens of thousands) that a single vectorized pass over all users is
    faster to implement and reason about than building/maintaining an index,
    while remaining well under interactive-latency budgets.
    """

    def __init__(self, users: list[UserFeatures], pool_size: int = 100):
        self.users = users
        self.pool_size = pool_size
        self._interest_matrix = np.stack([u.interest_vector for u in users])
        self._ages = np.array([u.age for u in users], dtype=np.float32)
        self._cities = np.array([u.city for u in users])
        self._countries = np.array([u.country for u in users])
        self._genders = np.array([u.gender for u in users])
        self._id_to_index = {u.user_id: i for i, u in enumerate(users)}

    def retrieve_indices(self, target: UserFeatures) -> np.ndarray:
        """Same scoring as retrieve(), but returns pool indices instead of UserFeatures
        objects -- lets callers pull batched feature rows straight from the
        precomputed matrices instead of rebuilding per-user vectors in Python."""
        target_vec = target.interest_vector.astype(bool)
        pool_vec = self._interest_matrix.astype(bool)

        intersection = np.logical_and(pool_vec, target_vec).sum(axis=1)
        union = np.logical_or(pool_vec, target_vec).sum(axis=1)
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union != 0)

        same_country = (self._countries == target.country).astype(np.float32)
        same_city = (self._cities == target.city).astype(np.float32)
        age_gap = np.abs(self._ages - target.age)
        age_score = 1.0 / (1.0 + age_gap / 10.0)

        score = (
            2.0 * jaccard
            + 1.0 * same_country
            + 1.5 * same_city
            + 0.5 * age_score
        )

        target_index = self._id_to_index.get(target.user_id)
        if target_index is not None:
            score[target_index] = -np.inf

        top_k = min(self.pool_size, len(self.users) - (1 if target_index is not None else 0))
        top_indices = np.argpartition(-score, top_k - 1)[:top_k]
        return top_indices[np.argsort(-score[top_indices])]

    def retrieve(self, target: UserFeatures) -> list[UserFeatures]:
        top_indices = self.retrieve_indices(target)
        return [self.users[i] for i in top_indices]

    def population_arrays(self) -> PopulationArrays:
        """Stable, public accessor for the retriever's precomputed population arrays."""
        return PopulationArrays(
            interest_matrix=self._interest_matrix,
            ages=self._ages,
            cities=self._cities,
            countries=self._countries,
            genders=self._genders,
        )
