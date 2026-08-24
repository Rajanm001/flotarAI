import torch

from app.models.ranker import PairwiseRanker
from app.services.features import RANKER_INPUT_DIM


def test_ranker_forward_shape():
    model = PairwiseRanker(input_dim=RANKER_INPUT_DIM)
    batch = torch.randn(8, RANKER_INPUT_DIM)
    out = model(batch)
    assert out.shape == (8,)


def test_ranker_deterministic_in_eval_mode():
    model = PairwiseRanker(input_dim=RANKER_INPUT_DIM)
    model.eval()
    batch = torch.randn(4, RANKER_INPUT_DIM)
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    assert torch.allclose(out1, out2)
