import time
from typing import Optional

from PIL import Image

from ..config import Settings
from ..models.schemas import AnalyzeRequest, AnalyzeResponse, ExifData
from .clip_embedder import classify_scene_tags, get_image_embedding
from .exif_extractor import extract_exif, load_image_from_url
from .plate_reader import read_license_plate
from .vehicle_detector import detect_vehicle


def analyze_image(request: AnalyzeRequest, settings: Settings) -> AnalyzeResponse:
    """
    Orchestrate the full AI analysis pipeline for a single image URL.

    Steps:
    1. Download image from URL
    2. Extract EXIF metadata (GPS, timestamp, camera model)
    3. Detect dominant vehicle type with YOLOv8
    4. Read license plate text with EasyOCR
    5. Generate 512-dim CLIP embedding
    6. Classify scene tags via CLIP zero-shot
    """
    start_ms = time.monotonic()

    response = AnalyzeResponse(moment_id=request.moment_id)

    try:
        image_bytes, image = load_image_from_url(request.image_url)
    except Exception as exc:
        response.error = f"Failed to load image: {exc}"
        response.processing_time_ms = int((time.monotonic() - start_ms) * 1000)
        return response

    # EXIF (fast, pure Python — always run)
    try:
        response.exif = extract_exif(image_bytes)
    except Exception:
        response.exif = ExifData()

    # Vehicle detection
    try:
        vehicle_type, vehicle_conf = detect_vehicle(
            image,
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.vehicle_confidence_threshold,
            vehicle_class_ids=settings.vehicle_class_ids,
        )
        response.vehicle_type = vehicle_type
        response.vehicle_confidence = vehicle_conf
    except Exception as exc:
        response.error = (response.error or "") + f" Vehicle detection failed: {exc}."

    # License plate OCR
    try:
        plate, plate_conf = read_license_plate(
            image,
            languages=settings.ocr_languages,
            confidence_threshold=settings.plate_confidence_threshold,
        )
        response.license_plate = plate
        response.plate_confidence = plate_conf
    except Exception as exc:
        response.error = (response.error or "") + f" Plate OCR failed: {exc}."

    # CLIP embedding
    try:
        response.embedding = get_image_embedding(image, model_name=settings.clip_model_name)
    except Exception as exc:
        response.error = (response.error or "") + f" CLIP embedding failed: {exc}."

    # Scene tag classification
    try:
        response.detected_tags = classify_scene_tags(image, model_name=settings.clip_model_name)
    except Exception as exc:
        response.error = (response.error or "") + f" Tag classification failed: {exc}."

    response.processing_time_ms = int((time.monotonic() - start_ms) * 1000)
    return response
