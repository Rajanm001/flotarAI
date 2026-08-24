"""
Offline evaluation on the held-out test split: for each test target user,
runs Stage A retrieval then Stage B ranking, computes ranking metrics against
the implicit relevance label, and reports candidate-generation recall and
end-to-end latency. Writes a summary to model_output/eval_metrics.json.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.core.logging import configure_logging
from app.models.ranker import PairwiseRanker
from app.services.features import UserFeatures
from app.services.labeling import is_relevant_batch, relevance_signals
from app.services.metrics import ranking_metrics
from app.services.ranker_inference import load_ranker, score_candidates
from app.services.retrieval import CandidateRetriever, PopulationArrays
from app.services.user_store import load_split_users

logger = logging.getLogger(__name__)


def load_test_users(max_test_targets: int, seed: int) -> tuple[dict[int, UserFeatures], list[int]]:
    all_users, split = load_split_users()

    test_ids = split["test"]
    rng = np.random.default_rng(seed)
    if len(test_ids) > max_test_targets:
        test_ids = [test_ids[i] for i in rng.choice(len(test_ids), max_test_targets, replace=False)]
    return all_users, test_ids


@dataclass(frozen=True)
class UserEvalResult:
    ranking_metrics: dict[str, float]
    latency_ms: float
    retrieval_recall: float | None


def evaluate_one_user(
    target: UserFeatures,
    retriever: CandidateRetriever,
    model: PairwiseRanker,
    population: PopulationArrays,
    population_interest_bool: np.ndarray,
    label_rng: np.random.Generator,
) -> UserEvalResult:
    t0 = time.perf_counter()
    pool_indices = retriever.retrieve_indices(target)
    candidates = [retriever.users[i] for i in pool_indices]
    scores = score_candidates(model, target, candidates)
    latency_ms = (time.perf_counter() - t0) * 1000

    order = np.argsort(-scores)
    ranked_candidates = [candidates[i] for i in order]
    ranked_interest_bool = np.stack([c.interest_vector for c in ranked_candidates]).astype(bool)
    ranked_cities = np.array([c.city for c in ranked_candidates])
    ranked_countries = np.array([c.country for c in ranked_candidates])
    ranked_ages = np.array([c.age for c in ranked_candidates], dtype=np.float32)

    jaccard, same_country, same_city, age_diff, rare_bonus = relevance_signals(
        target, ranked_interest_bool, ranked_cities, ranked_countries, ranked_ages
    )
    relevant = is_relevant_batch(jaccard, same_country, same_city, age_diff, rare_bonus, label_rng)
    metrics = ranking_metrics(relevant.astype(np.float32), k=settings.final_recommendation_count)

    # Candidate-generation recall: of all "relevant" users across the entire
    # dataset (not just the retrieved pool), what fraction did Stage A
    # actually surface into its top-100? Measured against the whole
    # population, unlike the ranking metrics above which are pool-scoped.
    full_jaccard, full_same_country, full_same_city, full_age_diff, full_rare_bonus = relevance_signals(
        target, population_interest_bool, population.cities, population.countries, population.ages
    )
    full_relevant = is_relevant_batch(
        full_jaccard, full_same_country, full_same_city, full_age_diff, full_rare_bonus, label_rng
    )
    total_relevant_in_dataset = int(full_relevant.sum())
    retrieval_recall = None
    if total_relevant_in_dataset > 0:
        relevant_in_pool = int(full_relevant[pool_indices].sum())
        retrieval_recall = relevant_in_pool / total_relevant_in_dataset

    return UserEvalResult(ranking_metrics=metrics, latency_ms=latency_ms, retrieval_recall=retrieval_recall)


def main(max_test_targets: int = 1000, seed: int = 123) -> None:
    configure_logging()
    all_users, test_ids = load_test_users(max_test_targets, seed)
    all_user_list = list(all_users.values())
    retriever = CandidateRetriever(all_user_list, pool_size=settings.candidate_pool_size)
    population = retriever.population_arrays()
    population_interest_bool = population.interest_matrix.astype(bool)

    model = load_ranker()
    label_rng = np.random.default_rng(seed)

    results = [
        evaluate_one_user(all_users[uid], retriever, model, population, population_interest_bool, label_rng)
        for uid in test_ids
    ]

    metric_keys = results[0].ranking_metrics.keys()
    aggregated = {
        key: float(np.mean([r.ranking_metrics[key] for r in results])) for key in metric_keys
    }
    latencies_ms = [r.latency_ms for r in results]
    recalls = [r.retrieval_recall for r in results if r.retrieval_recall is not None]

    aggregated["mean_latency_ms"] = float(np.mean(latencies_ms))
    aggregated["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
    aggregated["candidate_generation_recall"] = float(np.mean(recalls)) if recalls else None
    aggregated["num_test_users_evaluated"] = len(test_ids)

    logger.info("Evaluation results: %s", json.dumps(aggregated, indent=2))

    settings.model_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.model_output_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(aggregated, indent=2))
    logger.info("Wrote metrics to %s", out_path)


if __name__ == "__main__":
    main()
