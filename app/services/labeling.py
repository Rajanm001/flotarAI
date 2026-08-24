from __future__ import annotations

import numpy as np

from app.services.features import INTEREST_VOCAB, UserFeatures, jaccard_similarity

# Interest category frequencies observed in the full dataset (see
# scripts/prepare_data.py / notebook exploration). Used to weight shared
# *rare* interests more heavily than shared common ones -- e.g. two people
# both listing "Politics" is a stronger affinity signal than both listing
# "Travel", since almost everyone lists Travel. This rarity weighting is
# deliberately NOT exposed to the model as an input feature (the model only
# sees aggregate jaccard/shared_interest_count), so the label carries a bit
# of information the ranker has to approximate rather than read off directly.
_RARE_INTEREST_WEIGHT = {
    "Politics": 1.8, "Science": 1.6, "Social causes and activism": 1.6,
    "History": 1.4, "Gardening": 1.4, "DIY and crafts": 1.3,
    "Business and entrepreneurship": 1.3, "Parenting and family": 1.2,
    "Cars and automobiles": 1.2, "Photography": 1.1, "Art": 1.1,
    "Beauty": 1.0, "Fashion": 1.0, "Cooking": 1.0, "Health and wellness": 1.0,
    "Finance and investments": 1.0, "Nature": 1.0, "Education and learning": 1.0,
    "Pets": 0.9, "Fitness": 0.9, "Food and dining": 0.9, "Books": 0.9,
    "Outdoor activities": 0.8, "Sports": 0.8, "Gaming": 0.8, "Movies": 0.8,
    "Technology": 0.8, "Music": 0.7, "Travel": 0.7,
}
RARITY_VECTOR = np.array([_RARE_INTEREST_WEIGHT[name] for name in INTEREST_VOCAB], dtype=np.float32)


def rare_overlap_bonus(target_vec: np.ndarray, candidate_vec: np.ndarray) -> np.ndarray | float:
    """Sum of rarity weights over interests both users share."""
    shared = target_vec.astype(bool) & candidate_vec.astype(bool)
    return (shared.astype(np.float32) * RARITY_VECTOR).sum(axis=-1)


def latent_affinity_logit(
    jaccard: np.ndarray,
    same_country: np.ndarray,
    same_city: np.ndarray,
    age_diff: np.ndarray,
    rare_bonus: np.ndarray,
) -> np.ndarray:
    """
    Linear combination producing the logit of a Bernoulli "relevance"
    distribution used only to generate training/eval labels. This mixes the
    same coarse signals the retrieval/ranker features expose (jaccard,
    location, age) with the rare-interest bonus that is withheld from the
    model's inputs, and is deliberately not a hard threshold on any single
    input -- it is squashed through a sigmoid and sampled, so the model must
    learn an approximation of a noisy affinity function rather than an
    if-statement over its own features.
    """
    age_closeness = 1.0 / (1.0 + age_diff / 10.0)
    return (
        2.4 * jaccard
        + 0.9 * rare_bonus
        + 0.5 * same_country
        + 0.4 * same_city
        + 0.6 * age_closeness
        - 5.0
    )


def is_relevant(target: UserFeatures, candidate: UserFeatures, rng: np.random.Generator) -> bool:
    jaccard = jaccard_similarity(target.interest_vector, candidate.interest_vector)
    rare_bonus = float(rare_overlap_bonus(target.interest_vector, candidate.interest_vector))
    logit = latent_affinity_logit(
        jaccard=np.array(jaccard),
        same_country=np.array(1.0 if target.country == candidate.country else 0.0),
        same_city=np.array(1.0 if target.city == candidate.city else 0.0),
        age_diff=np.array(abs(target.age - candidate.age)),
        rare_bonus=np.array(rare_bonus),
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    return bool(rng.random() < prob)


def is_relevant_batch(
    jaccard: np.ndarray,
    same_country: np.ndarray,
    same_city: np.ndarray,
    age_diff: np.ndarray,
    rare_bonus: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    logit = latent_affinity_logit(jaccard, same_country, same_city, age_diff, rare_bonus)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return rng.random(size=prob.shape) < prob


def relevance_signals(
    target: UserFeatures,
    candidate_interest_bool: np.ndarray,
    candidate_cities: np.ndarray,
    candidate_countries: np.ndarray,
    candidate_ages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single vectorized source of truth for (jaccard, same_country, same_city,
    age_diff, rare_bonus) against a target user, given a candidate population
    already cast to bool interest vectors. Used identically whether the
    "candidate population" is a retrieved pool (~100 rows) or the entire
    dataset (~25k rows) -- callers differ only in which arrays they pass in,
    not in how the signals are computed, so the two can't silently drift
    apart the way two independently hand-written copies could.
    """
    target_interest = target.interest_vector.astype(bool)
    intersection = np.logical_and(candidate_interest_bool, target_interest).sum(axis=1)
    union = np.logical_or(candidate_interest_bool, target_interest).sum(axis=1)
    jaccard = np.divide(
        intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union != 0
    )
    same_country = (candidate_countries == target.country).astype(np.float32)
    same_city = (candidate_cities == target.city).astype(np.float32)
    age_diff = np.abs(candidate_ages - target.age).astype(np.float32)
    rare_bonus = (
        np.logical_and(candidate_interest_bool, target_interest).astype(np.float32) @ RARITY_VECTOR
    )
    return jaccard, same_country, same_city, age_diff, rare_bonus


def relevance_probability_batch(
    jaccard: np.ndarray,
    same_country: np.ndarray,
    same_city: np.ndarray,
    age_diff: np.ndarray,
    rare_bonus: np.ndarray,
) -> np.ndarray:
    logit = latent_affinity_logit(jaccard, same_country, same_city, age_diff, rare_bonus)
    return 1.0 / (1.0 + np.exp(-logit))
