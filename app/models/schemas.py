from typing import Optional
from pydantic import BaseModel, HttpUrl


class AnalyzeRequest(BaseModel):
    image_url: str
    moment_id: str


class ExifData(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    captured_at: Optional[int] = None
    camera_model: Optional[str] = None
    camera_make: Optional[str] = None


class AnalyzeResponse(BaseModel):
    moment_id: str
    vehicle_type: Optional[str] = None
    vehicle_confidence: Optional[float] = None
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    embedding: Optional[list[float]] = None
    detected_tags: list[str] = []
    exif: ExifData = ExifData()
    processing_time_ms: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]
