from __future__ import annotations

import torch
from torch import nn

from app.services.features import RANKER_INPUT_DIM


class PairwiseRanker(nn.Module):
    """
    MLP that scores affinity between a target user and one candidate.

    Input is [target_features | candidate_features | pairwise_features]
    concatenated per pair. Chosen over a Two-Tower architecture because the
    feature set is small and fully interpretable (interest overlap, location,
    age, gender) rather than learned embeddings over a large item catalog —
    a Two-Tower's main benefit (precomputing item embeddings for ANN search
    over millions of items) doesn't pay off at this dataset size, and a
    plain MLP over hand-engineered pairwise features is far easier to justify
    feature-by-feature in the README and interview.
    """

    def __init__(self, input_dim: int = RANKER_INPUT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
