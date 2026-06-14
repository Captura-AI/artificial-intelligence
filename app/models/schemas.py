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


class VehicleDetection(BaseModel):
    vehicle_type: str
    vehicle_confidence: float
    bbox: list[int]
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_image_path: Optional[str] = None


class AnalyzeResponse(BaseModel):
    moment_id: str
    vehicles: list[VehicleDetection] = []
    saved_photo: Optional[str] = None
    embedding: Optional[list[float]] = None
    detected_tags: list[str] = []
    exif: ExifData = ExifData()
    processing_time_ms: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]


class MotorDetection(BaseModel):
    motor_type: str
    motor_type_confidence: float
    color: Optional[str] = None
    color_confidence: Optional[float] = None
    plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    bbox: list[int]


class PlateScanResponse(BaseModel):
    uploader_id: str
    confidence: Optional[float] = None
    plates: list[str] = []
    motors: list[MotorDetection] = []
    saved_photo: Optional[str] = None
    saved_result_photo: Optional[str] = None
    error: Optional[str] = None


class PlateConfirmRequest(BaseModel):
    uploader_id: str
    action: str
    saved_photo_filename: Optional[str] = None
    saved_result_photo_filename: Optional[str] = None


class PlateConfirmResponse(BaseModel):
    uploader_id: str
    action: str
    success: bool
    message: str
