from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_dataset_path: Path = project_root / "data" / "Assessment_TwitterDataset.csv"
    processed_dir: Path = project_root / "data" / "processed"
    artifacts_dir: Path = project_root / "artifacts"

    model_filename: str = "ranker.pt"
    vocab_filename: str = "interest_vocab.json"

    random_seed: int = 42

    candidate_pool_size: int = 100
    final_recommendation_count: int = 10

    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    class Config:
        env_prefix = "FLOTER_"


settings = Settings()
