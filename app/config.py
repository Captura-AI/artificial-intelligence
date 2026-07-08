from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Captura AI Service"
    app_version: str = "1.0.0"
    debug: bool = False

    # API security. When `api_key` is set, protected endpoints require a matching
    # `X-API-Key` header; when left None, auth is disabled (convenient for local
    # dev). `cors_allow_origins` overrides the wildcard default in production.
    api_key: Optional[str] = None
    cors_allow_origins: list[str] = ["*"]

    # Model configuration
    yolo_model_path: str = "yolov8n.pt"
    clip_model_name: str = "ViT-B/32"
    vehicle_confidence_threshold: float = 0.5
    # Cosine-similarity cutoff for CLIP zero-shot scene tagging.
    clip_tag_threshold: float = 0.2

    # Plate detection / reading models
    platdetect_model_path: str = "app/aimodels/platdetect.pt"
    platreader_model_path: str = "app/aimodels/platreader.pt"
    platdetect_confidence_threshold: float = 0.25
    plate_padding_px: int = 15
    plate_confidence_threshold: float = 0.3

    # Motorcycle type detection + color classification models
    motortype_model_path: str = "app/aimodels/motortype.pt"
    color_model_path: str = "app/aimodels/color.pt"
    motortype_confidence_threshold: float = 0.4
    # Lower bar used only when a detected plate sits inside the motorcycle box.
    # A plate corroborates that a vehicle is present, so a bike that scores below
    # the standalone threshold can still be recovered into a complete record.
    motortype_assist_confidence_threshold: float = 0.2

    # YOLO class IDs that correspond to vehicles
    # COCO dataset: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_class_ids: list[int] = [1, 2, 3, 5, 7]

    # Supported languages for the legacy OCR reader
    ocr_languages: list[str] = ["en", "id"]

    # Local storage for temporary annotated plate scan output
    plate_save_dir: str = "temp_image"
    photo_save_dir: str = "saved_photos"
    annotated_photo_save_dir: str = "saved_results"

    # Max heavy pipeline runs (/analyze) allowed at once. A defense-in-depth cap
    # on top of the backend queue so the single-process model service is never
    # overloaded by parallel callers. Set to CPU/GPU headroom; 1 = fully serial.
    max_concurrent_analyses: int = 2

    # PostgreSQL connection string — override via DATABASE_URL in .env
    database_url: str = "postgresql://postgres:postgres@localhost:5432/captura"

    # Connection pool sizing. max_size should comfortably exceed the number of
    # threads that touch the DB at once (concurrent analyses + plate scans).
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
