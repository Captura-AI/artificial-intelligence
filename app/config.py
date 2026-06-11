from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Captura AI Service"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model configuration
    yolo_model_path: str = "yolov8n.pt"
    clip_model_name: str = "ViT-B/32"
    vehicle_confidence_threshold: float = 0.5
    plate_confidence_threshold: float = 0.4

    # Plate detection / reading models
    platdetect_model_path: str = "app/models/platdetect.pt"
    platreader_model_path: str = "app/models/platreader.pt"
    platdetect_confidence_threshold: float = 0.25
    plate_padding_px: int = 15
    plate_confidence_threshold: float = 0.3

    # Motorcycle type detection + color classification models
    motortype_model_path: str = "app/aimodels/motortype.pt"
    color_model_path: str = "app/aimodels/color.pt"
    motortype_confidence_threshold: float = 0.4

    # YOLO class IDs that correspond to vehicles
    # COCO dataset: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_class_ids: list[int] = [1, 2, 3, 5, 7]

    # Supported languages for the legacy OCR reader
    ocr_languages: list[str] = ["en", "id"]

    # Local storage for temporary annotated plate scan output
    plate_save_dir: str = "temp_image"
    photo_save_dir: str = "saved_photos"
    annotated_photo_save_dir: str = "saved_results"

    # PostgreSQL connection string — override via DATABASE_URL in .env
    database_url: str = "postgresql://postgres:postgres@localhost:5432/captura"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
