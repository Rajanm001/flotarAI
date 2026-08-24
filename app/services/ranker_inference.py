from __future__ import annotations

import numpy as np
import torch

from app.core.config import settings
from app.models.ranker import PairwiseRanker
from app.services.features import UserFeatures, pairwise_features, user_feature_vector


def load_ranker(path=None) -> PairwiseRanker:
    path = path or (settings.artifacts_dir / settings.model_filename)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = PairwiseRanker(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def score_candidates(
    model: PairwiseRanker, target: UserFeatures, candidates: list[UserFeatures]
) -> np.ndarray:
    """Batched affinity scoring for all candidates against one target user."""
    target_vec = user_feature_vector(target)
    rows = []
    for candidate in candidates:
        candidate_vec = user_feature_vector(candidate)
        pair_vec = pairwise_features(target, candidate)
        rows.append(np.concatenate([target_vec, candidate_vec, pair_vec]))

    batch = torch.from_numpy(np.stack(rows).astype(np.float32))
    with torch.no_grad():
        logits = model(batch)
    return torch.sigmoid(logits).numpy()


def rank_top_k(
    model: PairwiseRanker, target: UserFeatures, candidates: list[UserFeatures], k: int
) -> list[tuple[UserFeatures, float]]:
    scores = score_candidates(model, target, candidates)
    order = np.argsort(-scores)[:k]
    return [(candidates[i], float(scores[i])) for i in order]
