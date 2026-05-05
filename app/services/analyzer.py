import time
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from ..config import Settings
from ..db.crud import save_plate_result
from ..models.schemas import AnalyzeRequest, AnalyzeResponse, ExifData
from .clip_embedder import classify_scene_tags, get_image_embedding
from .exif_extractor import extract_exif, load_image_from_url
from .plate_detector import detect_plates
from .plate_text_reader import read_plate_text
from .vehicle_detector import detect_vehicle


def _save_plate_image(plate_crop: Image.Image, save_dir: str, moment_id: str) -> str:
    """
    Save a cropped plate image to disk and return its absolute file path.
    Directory is created on demand; file is named <moment_id>_<uuid>.jpg.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{moment_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir / filename)
    plate_crop.convert("RGB").save(file_path, format="JPEG")
    return file_path


def analyze_image(request: AnalyzeRequest, settings: Settings) -> AnalyzeResponse:
    """
    Orchestrate the full AI analysis pipeline for a single image URL.

    Steps:
    1. Download image from URL
    2. Extract EXIF metadata (GPS, timestamp, camera model)
    3. Detect dominant vehicle type with YOLOv8
    4. Detect plate region with platdetect.pt
    5. Read plate text with platreader.pt
    6. Save plate crop to disk; record path + text in SQLite
    7. Generate 512-dim CLIP embedding
    8. Classify scene tags via CLIP zero-shot
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
    detected_vehicles = []
    try:
        detected_vehicles = detect_vehicle(
            image,
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.vehicle_confidence_threshold,
            vehicle_class_ids=settings.vehicle_class_ids,
        )
    except Exception as exc:
        response.error = (response.error or "") + f" Vehicle detection failed: {exc}."

    # Iterate over all detected vehicles — detect plate then read text
    from ..models.schemas import VehicleDetection

    for v_type, v_conf, v_bbox in detected_vehicles:
        vehicle_data = VehicleDetection(
            vehicle_type=v_type,
            vehicle_confidence=v_conf,
            bbox=v_bbox,
        )

        try:
            vehicle_crop = image.crop((v_bbox[0], v_bbox[1], v_bbox[2], v_bbox[3]))

            # Step 1 — locate the plate region inside the vehicle crop
            plate_boxes = detect_plates(
                vehicle_crop,
                model_path=settings.platdetect_model_path,
                confidence_threshold=settings.platdetect_confidence_threshold,
            )

            plate_crop: Optional[Image.Image] = None
            if plate_boxes:
                # Use the highest-confidence plate detection
                _, p_bbox = plate_boxes[0]
                plate_crop = vehicle_crop.crop(
                    (p_bbox[0], p_bbox[1], p_bbox[2], p_bbox[3])
                )
            else:
                # Fall back to the full vehicle crop when no plate is detected
                plate_crop = vehicle_crop

            # Step 2 — read text from the plate crop
            plate_text, plate_conf = read_plate_text(
                plate_crop,
                model_path=settings.platreader_model_path,
                confidence_threshold=settings.plate_confidence_threshold,
            )

            if plate_text:
                # Step 3 — save the plate image to disk
                saved_path = _save_plate_image(
                    plate_crop, settings.plate_save_dir, request.moment_id
                )
                vehicle_data.license_plate = plate_text
                vehicle_data.plate_confidence = plate_conf
                vehicle_data.plate_image_path = saved_path

                # Step 4 — persist path + text to the database
                save_plate_result(
                    database_url=settings.database_url,
                    moment_id=request.moment_id,
                    file_path=saved_path,
                    plate_text=plate_text,
                    confidence=plate_conf,
                )

        except Exception as exc:
            response.error = (
                response.error or ""
            ) + f" Plate pipeline failed for vehicle {v_type}: {exc}."

        response.vehicles.append(vehicle_data)

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
