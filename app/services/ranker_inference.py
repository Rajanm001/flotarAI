from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.core.config import settings
from app.models.ranker import PairwiseRanker
from app.services.features import (
    UserFeatures,
    batch_pairwise_features,
    batch_user_feature_matrix,
)


def load_ranker(path: Path | None = None) -> PairwiseRanker:
    path = path or (settings.model_output_dir / settings.model_filename)
    # weights_only=True restricts deserialization to plain tensors/primitives
    # (the checkpoint here only ever contains model_state/input_dim/val_loss),
    # so a corrupted or tampered .pt file fails safely instead of executing
    # arbitrary pickled objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PairwiseRanker(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def score_candidates(
    model: PairwiseRanker, target: UserFeatures, candidates: list[UserFeatures]
) -> np.ndarray:
    """Batched affinity scoring for all candidates against one target user."""
    if not candidates:
        return np.empty(0, dtype=np.float32)

    target_vec = batch_user_feature_matrix([target])[0]
    candidate_matrix = batch_user_feature_matrix(candidates)
    pairwise_matrix = batch_pairwise_features(target, candidates)

    target_block = np.tile(target_vec, (len(candidates), 1))
    batch = np.concatenate([target_block, candidate_matrix, pairwise_matrix], axis=1).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(batch))
    return torch.sigmoid(logits).numpy()


def rank_top_k(
    model: PairwiseRanker, target: UserFeatures, candidates: list[UserFeatures], k: int
) -> list[tuple[UserFeatures, float]]:
    if not candidates:
        return []
    scores = score_candidates(model, target, candidates)
    order = np.argsort(-scores)[:k]
    return [(candidates[i], float(scores[i])) for i in order]
