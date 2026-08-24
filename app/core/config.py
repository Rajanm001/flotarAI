import datetime as dt
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLOTER_")

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_dataset_path: Path = project_root / "data" / "Assessment_TwitterDataset.csv"
    processed_dir: Path = project_root / "data" / "processed"
    artifacts_dir: Path = project_root / "artifacts"

    model_filename: str = "ranker.pt"

    random_seed: int = 42

    candidate_pool_size: int = 100
    final_recommendation_count: int = 10

    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # "As-of" date used to compute age from DOB. Pinned to a fixed date
    # (rather than the wall-clock "today") so age features -- and therefore
    # training data, evaluation metrics, and sample_results.csv -- are
    # exactly reproducible across repeated runs regardless of what day they
    # are run on. Override with FLOTER_AGE_REFERENCE_DATE if needed.
    age_reference_date: dt.date = dt.date(2026, 8, 27)


settings = Settings()
