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
from .alpr_pipeline import PlatePipelineResult, run_plate_alpr
from .motor_attribute_pipeline import MotorAttributeResult, run_motor_attributes
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


def _save_original_image(image: Image.Image, save_dir: str, moment_id: str) -> str:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{moment_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir / filename)
    image.convert("RGB").save(file_path, format="JPEG")
    return file_path


def _bbox_contains_point(bbox: list[int], x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _plate_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _match_motor_to_vehicle(
    vehicle_bbox: list[int],
    motor_results: list[MotorAttributeResult],
    used_motor_indexes: set[int],
    min_iou: float = 0.3,
) -> Optional[int]:
    """Return the index of the motor-attribute detection that best overlaps the
    vehicle box (highest IoU above ``min_iou``), so a MOTORCYCLE vehicle can be
    enriched with its body style + color."""
    best_index: Optional[int] = None
    best_iou = min_iou

    for index, motor_result in enumerate(motor_results):
        if index in used_motor_indexes:
            continue
        iou = _bbox_iou(vehicle_bbox, motor_result.bbox)
        if iou >= best_iou:
            best_index = index
            best_iou = iou

    return best_index


def _match_plate_to_vehicle(
    vehicle_bbox: list[int],
    plate_results: list[PlatePipelineResult],
    used_plate_indexes: set[int],
) -> Optional[int]:
    best_index: Optional[int] = None
    best_score: tuple[float, float] | None = None

    for index, plate_result in enumerate(plate_results):
        if index in used_plate_indexes or not plate_result.text:
            continue

        center_x, center_y = _plate_center(plate_result.bbox)
        if not _bbox_contains_point(vehicle_bbox, center_x, center_y):
            continue

        score = (
            plate_result.text_confidence or 0.0,
            plate_result.detection_confidence,
        )
        if best_score is None or score > best_score:
            best_index = index
            best_score = score

    return best_index


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

    try:
        response.saved_photo = _save_original_image(
            image, settings.photo_save_dir, request.moment_id
        )
    except Exception as exc:
        response.error = (response.error or "") + f" Saving original photo failed: {exc}."

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

    plate_results: list[PlatePipelineResult] = []
    saved_plate_paths: dict[int, str] = {}
    try:
        plate_results = run_plate_alpr(
            image=image,
            detector_model_path=settings.platdetect_model_path,
            reader_model_path=settings.platreader_model_path,
            detector_confidence_threshold=settings.platdetect_confidence_threshold,
            reader_confidence_threshold=settings.plate_confidence_threshold,
            padding_px=settings.plate_padding_px,
        )
    except Exception as exc:
        response.error = (response.error or "") + f" Plate pipeline failed: {exc}."

    # Motorcycle body style + color (independent of the plate stage). Detected
    # down to the assist threshold so a low-confidence bike confirmed by a plate
    # can still be enriched; standalone filtering happens via IoU matching below.
    motor_results: list[MotorAttributeResult] = []
    try:
        motor_results = run_motor_attributes(
            image=image,
            motortype_model_path=settings.motortype_model_path,
            color_model_path=settings.color_model_path,
            confidence_threshold=settings.motortype_assist_confidence_threshold,
        )
    except Exception as exc:
        response.error = (response.error or "") + f" Motor attribute pipeline failed: {exc}."

    used_plate_indexes: set[int] = set()
    used_motor_indexes: set[int] = set()

    # Iterate over all detected vehicles — match each vehicle with a full-image plate detection
    from ..models.schemas import VehicleDetection

    for v_type, v_conf, v_bbox in detected_vehicles:
        vehicle_data = VehicleDetection(
            vehicle_type=v_type,
            vehicle_confidence=v_conf,
            bbox=v_bbox,
        )

        if v_type == "MOTORCYCLE":
            matched_motor_index = _match_motor_to_vehicle(
                vehicle_bbox=v_bbox,
                motor_results=motor_results,
                used_motor_indexes=used_motor_indexes,
            )
            if matched_motor_index is not None:
                used_motor_indexes.add(matched_motor_index)
                matched_motor = motor_results[matched_motor_index]
                vehicle_data.motor_type = matched_motor.motor_type
                vehicle_data.motor_type_confidence = matched_motor.motor_type_confidence
                vehicle_data.color = matched_motor.color
                vehicle_data.color_confidence = matched_motor.color_confidence

        try:
            matched_plate_index = _match_plate_to_vehicle(
                vehicle_bbox=v_bbox,
                plate_results=plate_results,
                used_plate_indexes=used_plate_indexes,
            )

            if matched_plate_index is not None:
                used_plate_indexes.add(matched_plate_index)
                matched_plate = plate_results[matched_plate_index]

                saved_path = saved_plate_paths.get(matched_plate_index)
                if saved_path is None:
                    saved_path = _save_plate_image(
                        matched_plate.crop, settings.plate_save_dir, request.moment_id
                    )
                    saved_plate_paths[matched_plate_index] = saved_path
                    save_plate_result(
                        database_url=settings.database_url,
                        moment_id=request.moment_id,
                        file_path=saved_path,
                        plate_text=matched_plate.text,
                        confidence=matched_plate.text_confidence,
                    )

                vehicle_data.license_plate = matched_plate.text
                vehicle_data.plate_confidence = matched_plate.text_confidence
                vehicle_data.plate_image_path = saved_path

        except Exception as exc:
            response.error = (
                response.error or ""
            ) + f" Plate pipeline failed for vehicle {v_type}: {exc}."

        response.vehicles.append(vehicle_data)

    # Flatten the highest-confidence vehicle into top-level fields. detected_vehicles
    # is sorted by confidence descending, so the first appended vehicle is dominant.
    # The backend reads these scalars directly to auto-fill moment columns.
    if response.vehicles:
        dominant = response.vehicles[0]
        response.vehicle_type = dominant.vehicle_type
        response.vehicle_confidence = dominant.vehicle_confidence
        response.license_plate = dominant.license_plate
        response.plate_confidence = dominant.plate_confidence
        response.motor_type = dominant.motor_type
        response.color = dominant.color

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
