"""
Offline evaluation on the held-out test split: for each test target user,
runs Stage A retrieval then Stage B ranking, computes ranking metrics against
the implicit relevance label, and reports candidate-generation recall and
end-to-end latency. Writes a summary to artifacts/eval_metrics.json.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logging import configure_logging
from app.services.features import build_user_features
from app.services.labeling import RARITY_VECTOR, is_relevant_batch
from app.services.metrics import ranking_metrics
from app.services.ranker_inference import load_ranker, score_candidates
from app.services.retrieval import CandidateRetriever

logger = logging.getLogger(__name__)


def main(max_test_targets: int = 1000, seed: int = 123) -> None:
    configure_logging()
    df = pd.read_csv(settings.processed_dir / "users.csv")
    split = json.loads((settings.processed_dir / "user_split.json").read_text())

    all_users = {u.user_id: u for u in (build_user_features(row) for _, row in df.iterrows())}
    all_user_list = list(all_users.values())
    retriever = CandidateRetriever(all_user_list, pool_size=settings.candidate_pool_size)

    test_ids = split["test"]
    rng = np.random.default_rng(seed)
    if len(test_ids) > max_test_targets:
        test_ids = [test_ids[i] for i in rng.choice(len(test_ids), max_test_targets, replace=False)]

    model = load_ranker()

    label_rng = np.random.default_rng(seed)
    per_user_metrics = []
    retrieval_recalls = []
    latencies_ms = []

    candidate_matrix = retriever._interest_matrix
    candidate_ages = retriever._ages
    candidate_cities = retriever._cities
    candidate_countries = retriever._countries

    for uid in test_ids:
        target = all_users[uid]

        t0 = time.perf_counter()
        pool_indices = retriever.retrieve_indices(target)
        candidates = [all_user_list[i] for i in pool_indices]
        scores = score_candidates(model, target, candidates)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        order = np.argsort(-scores)
        ranked_candidates = [candidates[i] for i in order]

        target_interest = target.interest_vector.astype(bool)
        jaccard_arr = np.array([
            float(np.logical_and(target_interest, c.interest_vector.astype(bool)).sum())
            / max(1, np.logical_or(target_interest, c.interest_vector.astype(bool)).sum())
            for c in ranked_candidates
        ], dtype=np.float32)
        same_country_arr = np.array([1.0 if c.country == target.country else 0.0 for c in ranked_candidates], dtype=np.float32)
        same_city_arr = np.array([1.0 if c.city == target.city else 0.0 for c in ranked_candidates], dtype=np.float32)
        age_diff_arr = np.array([abs(c.age - target.age) for c in ranked_candidates], dtype=np.float32)
        rare_bonus_arr = np.array([
            float((np.logical_and(target_interest, c.interest_vector.astype(bool)).astype(np.float32) * RARITY_VECTOR).sum())
            for c in ranked_candidates
        ], dtype=np.float32)

        relevant = is_relevant_batch(jaccard_arr, same_country_arr, same_city_arr, age_diff_arr, rare_bonus_arr, label_rng)
        per_user_metrics.append(ranking_metrics(relevant.astype(np.float32), k=settings.final_recommendation_count))

        # Candidate-generation recall: of all "relevant" users across the
        # *entire* dataset (not just the retrieved pool), what fraction did
        # Stage A actually surface into its top-100? This is measured
        # against the whole population, unlike the ranking metrics above
        # which are computed only within the already-retrieved pool.
        full_jaccard = np.divide(
            np.logical_and(candidate_matrix.astype(bool), target_interest).sum(axis=1),
            np.logical_or(candidate_matrix.astype(bool), target_interest).sum(axis=1),
            out=np.zeros(len(all_user_list), dtype=np.float32),
            where=np.logical_or(candidate_matrix.astype(bool), target_interest).sum(axis=1) != 0,
        )
        full_same_country = (candidate_countries == target.country).astype(np.float32)
        full_same_city = (candidate_cities == target.city).astype(np.float32)
        full_age_diff = np.abs(candidate_ages - target.age).astype(np.float32)
        full_rare_bonus = (
            np.logical_and(candidate_matrix.astype(bool), target_interest).astype(np.float32) @ RARITY_VECTOR
        )
        full_relevant = is_relevant_batch(
            full_jaccard, full_same_country, full_same_city, full_age_diff, full_rare_bonus, label_rng
        )
        total_relevant_in_dataset = int(full_relevant.sum())
        relevant_in_pool = int(full_relevant[pool_indices].sum())
        if total_relevant_in_dataset > 0:
            retrieval_recalls.append(relevant_in_pool / total_relevant_in_dataset)

    metric_keys = per_user_metrics[0].keys()
    aggregated = {key: float(np.mean([m[key] for m in per_user_metrics])) for key in metric_keys}
    aggregated["mean_latency_ms"] = float(np.mean(latencies_ms))
    aggregated["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
    aggregated["candidate_generation_recall"] = float(np.mean(retrieval_recalls)) if retrieval_recalls else None
    aggregated["num_test_users_evaluated"] = len(test_ids)

    logger.info("Evaluation results: %s", json.dumps(aggregated, indent=2))

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.artifacts_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(aggregated, indent=2))
    logger.info("Wrote metrics to %s", out_path)


if __name__ == "__main__":
    main()
