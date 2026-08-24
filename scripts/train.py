"""
Trains the PairwiseRanker on (target, candidate) pairs drawn from each
target user's Stage A retrieval pool, using the implicit relevance label
from app.services.labeling. Reports validation metrics each epoch and saves
the best-validation-loss checkpoint to artifacts/.

Run `python scripts/prepare_data.py` first to generate the train/val/test
user id split this script reads from data/processed/user_split.json.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import settings
from app.core.logging import configure_logging
from app.models.ranker import PairwiseRanker
from app.services.pairs import build_training_pairs
from app.services.retrieval import CandidateRetriever
from app.services.user_store import load_split_users

logger = logging.getLogger(__name__)


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * len(y)
            total_count += len(y)
    return total_loss / total_count


def main(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_train_targets: int = 4000,
    max_val_targets: int = 1000,
) -> None:
    configure_logging()
    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    all_users, split = load_split_users()
    all_user_list = list(all_users.values())
    retriever = CandidateRetriever(all_user_list, pool_size=settings.candidate_pool_size)

    train_targets = [all_users[uid] for uid in split["train"]]
    val_targets = [all_users[uid] for uid in split["val"]]

    logger.info("Building training pairs for up to %d target users", max_train_targets)
    train_x, train_y = build_training_pairs(train_targets, retriever, max_targets=max_train_targets)
    logger.info("Building validation pairs for up to %d target users", max_val_targets)
    val_x, val_y = build_training_pairs(val_targets, retriever, max_targets=max_val_targets)

    logger.info(
        "train pairs: %d (pos rate %.3f), val pairs: %d (pos rate %.3f)",
        len(train_y), train_y.mean(), len(val_y), val_y.mean(),
    )

    train_loader = make_loader(train_x, train_y, batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, batch_size, shuffle=False)

    model = PairwiseRanker(input_dim=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.artifacts_dir / settings.model_filename

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            running_count += len(y)

        train_loss = running_loss / running_count
        val_loss = evaluate(model, val_loader, criterion)
        logger.info("epoch %d/%d train_loss=%.4f val_loss=%.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": train_x.shape[1],
                    "val_loss": val_loss,
                },
                model_path,
            )
            logger.info("Saved new best checkpoint (val_loss=%.4f) to %s", val_loss, model_path)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
