import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _build_default_sqlite_uri(db_name: str) -> str:
    db_path = PROJECT_ROOT / db_name
    return f"sqlite:///{db_path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    videos_api_key: str
    comments_api_key: str
    stats_api_key: str
    mlflow_tracking_uri: str
    mlflow_registry_uri: str
    reddit_dataset_path: str
    data_output_path: str
    prediction_logs_path: str
    baseline_dataset_path: str
    hand_labeled_test_path: str
    api_base_url: str
    promotion_metric: str
    min_hand_labeled_macro_f1: float
    min_negative_recall: float
    min_class_ratio: float
    require_probabilistic_champion: bool
    max_batch_size: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _build_default_sqlite_uri("mlflow.db"))
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    require_probabilistic_champion = os.getenv("REQUIRE_PROBABILISTIC_CHAMPION", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return Settings(
        videos_api_key=os.getenv("YOUTUBE_VIDEOS_API_KEY", ""),
        comments_api_key=os.getenv("YOUTUBE_COMMENTS_API_KEY", ""),
        stats_api_key=os.getenv("YOUTUBE_STATS_API_KEY", ""),
        mlflow_tracking_uri=tracking_uri,
        mlflow_registry_uri=registry_uri,
        reddit_dataset_path=os.getenv("REDDIT_DATASET_PATH", ""),
        data_output_path=os.getenv("DATA_OUTPUT_PATH", "sentiment_analysis_data.parquet"),
        prediction_logs_path=os.getenv("PREDICTION_LOGS_PATH", "prediction_logs.csv"),
        baseline_dataset_path=os.getenv("BASELINE_DATASET_PATH", "sentiment_analysis_data.parquet"),
        hand_labeled_test_path=os.getenv("HAND_LABELED_TEST_PATH", ""),
        api_base_url=os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8001"),
        promotion_metric=os.getenv("PROMOTION_METRIC", "macro_f1"),
        min_hand_labeled_macro_f1=float(os.getenv("MIN_HAND_LABELED_MACRO_F1", "0.65")),
        min_negative_recall=float(os.getenv("MIN_NEGATIVE_RECALL", "0.55")),
        min_class_ratio=float(os.getenv("MIN_CLASS_RATIO", "0.10")),
        require_probabilistic_champion=require_probabilistic_champion,
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
    )
