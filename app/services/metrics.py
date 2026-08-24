from __future__ import annotations

import numpy as np


def precision_at_k(relevant: np.ndarray, k: int) -> float:
    top_k = relevant[:k]
    return float(top_k.sum() / k) if k > 0 else 0.0


def recall_at_k(relevant: np.ndarray, k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    top_k = relevant[:k]
    return float(top_k.sum() / total_relevant)


def hit_rate_at_k(relevant: np.ndarray, k: int) -> float:
    return 1.0 if relevant[:k].sum() > 0 else 0.0


def ndcg_at_k(relevant: np.ndarray, k: int) -> float:
    top_k = relevant[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(top_k) + 2))
    dcg = float((top_k * discounts).sum())

    ideal = np.sort(top_k)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def ranking_metrics(sorted_labels: np.ndarray, k: int = 10) -> dict[str, float]:
    total_relevant = int(sorted_labels.sum())
    return {
        f"precision_at_{k}": precision_at_k(sorted_labels, k),
        f"recall_at_{k}": recall_at_k(sorted_labels, k, total_relevant),
        f"hit_rate_at_{k}": hit_rate_at_k(sorted_labels, k),
        f"ndcg_at_{k}": ndcg_at_k(sorted_labels, k),
    }
