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

    # YOLO class IDs that correspond to vehicles
    # COCO dataset: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_class_ids: list[int] = [1, 2, 3, 5, 7]

    # Supported languages for OCR (Indonesian plates)
    ocr_languages: list[str] = ["en", "id"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
