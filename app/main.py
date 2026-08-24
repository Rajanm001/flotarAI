from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.services.ranker_inference import load_ranker
from app.services.user_store import UserStore

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading user store and ranker model...")
    if not settings.raw_dataset_path.exists():
        raise RuntimeError(
            f"Dataset not found at {settings.raw_dataset_path}. "
            "Place Assessment_TwitterDataset.csv in data/ before starting the API."
        )
    model_path = settings.model_output_dir / settings.model_filename
    if not model_path.exists():
        raise RuntimeError(
            f"Trained model not found at {model_path}. Run `python -m scripts.train` first."
        )

    try:
        app.state.user_store = UserStore.from_csv()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load user pool from {settings.raw_dataset_path}: {exc}"
        ) from exc

    try:
        app.state.ranker_model = load_ranker(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load trained model from {model_path}: {exc}") from exc

    logger.info("Startup complete.")
    yield


app = FastAPI(title="Floter Recommendation Engine", lifespan=lifespan)
app.include_router(router)
