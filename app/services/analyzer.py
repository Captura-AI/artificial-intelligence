import logging
import time
from typing import Optional

from ..config import Settings
from ..db.crud import save_plate_result
from ..models.schemas import AnalyzeRequest, AnalyzeResponse, ExifData
from .alpr_pipeline import PlatePipelineResult, run_plate_alpr
from .clip_embedder import classify_scene_tags, get_image_embedding
from .exif_extractor import extract_exif, load_image_from_url
from .geometry import _bbox_contains_point, _bbox_iou, _plate_center
from .motor_attribute_pipeline import MotorAttributeResult, run_motor_attributes
from .utils import save_jpeg
from .vehicle_detector import detect_vehicle

logger = logging.getLogger(__name__)


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
    4. Detect plate region with platdetect.pt + read text with platreader.pt
    5. Match plates to vehicles (vehicle-gated); save unmatched plates separately
    6. Motorcycle body style + color classification
    7. Generate 512-dim CLIP embedding
    8. Classify scene tags via CLIP zero-shot
    """
    start_ms = time.monotonic()
    mid = request.moment_id
    response = AnalyzeResponse(moment_id=mid)

    logger.info("analyze_image.start moment_id=%s image_url=%s", mid, request.image_url)

    # ── Image download ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        image_bytes, image = load_image_from_url(request.image_url)
        logger.debug(
            "analyze_image.load_image done ms=%d", int((time.monotonic() - t0) * 1000)
        )
    except Exception as exc:
        logger.error("analyze_image.load_image failed moment_id=%s error=%s", mid, exc)
        response.error = f"Failed to load image: {exc}"
        response.processing_time_ms = int((time.monotonic() - start_ms) * 1000)
        return response

    try:
        response.saved_photo = save_jpeg(image, settings.photo_save_dir, mid)
    except Exception as exc:
        logger.warning(
            "analyze_image.save_original failed moment_id=%s error=%s", mid, exc
        )
        response.error = (
            response.error or ""
        ) + f" Saving original photo failed: {exc}."

    # ── EXIF ──────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        response.exif = extract_exif(image_bytes)
        logger.debug(
            "analyze_image.exif done ms=%d gps=(%s,%s) camera=%s",
            int((time.monotonic() - t0) * 1000),
            response.exif.latitude,
            response.exif.longitude,
            response.exif.camera_model,
        )
    except Exception:
        response.exif = ExifData()

    # ── Vehicle detection ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    detected_vehicles = []
    try:
        detected_vehicles = detect_vehicle(
            image,
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.vehicle_confidence_threshold,
            vehicle_class_ids=settings.vehicle_class_ids,
        )
        logger.debug(
            "analyze_image.vehicle_detection done ms=%d count=%d types=%s",
            int((time.monotonic() - t0) * 1000),
            len(detected_vehicles),
            [v[0] for v in detected_vehicles],
        )
    except Exception as exc:
        logger.error(
            "analyze_image.vehicle_detection failed moment_id=%s error=%s", mid, exc
        )
        response.error = (response.error or "") + f" Vehicle detection failed: {exc}."

    # ── Plate ALPR ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
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
        readable = [p for p in plate_results if p.text]
        logger.debug(
            "analyze_image.plate_alpr done ms=%d detected=%d readable=%d texts=%s",
            int((time.monotonic() - t0) * 1000),
            len(plate_results),
            len(readable),
            [p.text for p in readable],
        )
    except Exception as exc:
        logger.error("analyze_image.plate_alpr failed moment_id=%s error=%s", mid, exc)
        response.error = (response.error or "") + f" Plate pipeline failed: {exc}."

    # ── Motor attributes ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    motor_results: list[MotorAttributeResult] = []
    try:
        motor_results = run_motor_attributes(
            image=image,
            motortype_model_path=settings.motortype_model_path,
            color_model_path=settings.color_model_path,
            confidence_threshold=settings.motortype_assist_confidence_threshold,
        )
        logger.debug(
            "analyze_image.motor_attrs done ms=%d count=%d",
            int((time.monotonic() - t0) * 1000),
            len(motor_results),
        )
    except Exception as exc:
        logger.error("analyze_image.motor_attrs failed moment_id=%s error=%s", mid, exc)
        response.error = (
            response.error or ""
        ) + f" Motor attribute pipeline failed: {exc}."

    # ── Vehicle ↔ plate matching ───────────────────────────────────────────────
    from ..models.schemas import VehicleDetection

    used_plate_indexes: set[int] = set()
    used_motor_indexes: set[int] = set()

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
                    saved_path = save_jpeg(
                        matched_plate.crop, settings.plate_save_dir, mid
                    )
                    saved_plate_paths[matched_plate_index] = saved_path
                    save_plate_result(
                        database_url=settings.database_url,
                        moment_id=mid,
                        file_path=saved_path,
                        plate_text=matched_plate.text,
                        confidence=matched_plate.text_confidence,
                    )
                    logger.debug(
                        "analyze_image.plate_saved moment_id=%s plate=%s conf=%.2f vehicle=%s",
                        mid,
                        matched_plate.text,
                        matched_plate.text_confidence or 0.0,
                        v_type,
                    )

                vehicle_data.license_plate = matched_plate.text
                vehicle_data.plate_confidence = matched_plate.text_confidence
                vehicle_data.plate_image_path = saved_path

        except Exception as exc:
            logger.error(
                "analyze_image.plate_match failed moment_id=%s vehicle=%s error=%s",
                mid,
                v_type,
                exc,
            )
            response.error = (
                response.error or ""
            ) + f" Plate pipeline failed for vehicle {v_type}: {exc}."

        response.vehicles.append(vehicle_data)

    # ── Unmatched plates (plate detected, no vehicle bounding-box overlap) ────
    # Mirrors /plate/scan behaviour: persist plates that are not inside any
    # detected vehicle box so uploads without vehicle detection still get a
    # plate reading. This is the key parity fix: /analyze now returns a plate
    # even when YOLOv8 misses the vehicle.
    unmatched_readable = [
        (i, p)
        for i, p in enumerate(plate_results)
        if i not in used_plate_indexes and p.text
    ]
    for plate_index, unmatched_plate in unmatched_readable:
        try:
            saved_path = saved_plate_paths.get(plate_index)
            if saved_path is None:
                saved_path = save_jpeg(
                    unmatched_plate.crop, settings.plate_save_dir, mid
                )
                saved_plate_paths[plate_index] = saved_path
                save_plate_result(
                    database_url=settings.database_url,
                    moment_id=mid,
                    file_path=saved_path,
                    plate_text=unmatched_plate.text,
                    confidence=unmatched_plate.text_confidence,
                )
                logger.debug(
                    "analyze_image.unmatched_plate_saved moment_id=%s plate=%s conf=%.2f",
                    mid,
                    unmatched_plate.text,
                    unmatched_plate.text_confidence or 0.0,
                )
        except Exception as exc:
            logger.warning(
                "analyze_image.unmatched_plate_save_failed moment_id=%s plate=%s error=%s",
                mid,
                unmatched_plate.text,
                exc,
            )

    # ── Flatten dominant vehicle into top-level scalars ───────────────────────
    # The backend reads these directly to auto-fill moment columns.
    if response.vehicles:
        dominant = response.vehicles[0]
        response.vehicle_type = dominant.vehicle_type
        response.vehicle_confidence = dominant.vehicle_confidence
        response.license_plate = dominant.license_plate
        response.plate_confidence = dominant.plate_confidence
        response.motor_type = dominant.motor_type
        response.color = dominant.color
    elif unmatched_readable:
        # No vehicle detected, but at least one plate was read — surface the
        # best plate so the backend can still auto-fill licensePlate.
        best = max(unmatched_readable, key=lambda ip: ip[1].text_confidence or 0.0)[1]
        response.license_plate = best.text
        response.plate_confidence = best.text_confidence
        logger.info(
            "analyze_image.plate_without_vehicle moment_id=%s plate=%s conf=%.2f",
            mid,
            best.text,
            best.text_confidence or 0.0,
        )

    # ── CLIP embedding ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        response.embedding = get_image_embedding(
            image, model_name=settings.clip_model_name
        )
        logger.debug(
            "analyze_image.clip_embed done ms=%d", int((time.monotonic() - t0) * 1000)
        )
    except Exception as exc:
        logger.error("analyze_image.clip_embed failed moment_id=%s error=%s", mid, exc)
        response.error = (response.error or "") + f" CLIP embedding failed: {exc}."

    # ── Scene tags ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        response.detected_tags = classify_scene_tags(
            image, model_name=settings.clip_model_name
        )
        logger.debug(
            "analyze_image.scene_tags done ms=%d tags=%s",
            int((time.monotonic() - t0) * 1000),
            response.detected_tags,
        )
    except Exception as exc:
        logger.error("analyze_image.scene_tags failed moment_id=%s error=%s", mid, exc)
        response.error = (response.error or "") + f" Tag classification failed: {exc}."

    response.processing_time_ms = int((time.monotonic() - start_ms) * 1000)

    logger.info(
        "analyze_image.complete moment_id=%s total_ms=%d vehicles=%d plate=%s error=%s",
        mid,
        response.processing_time_ms,
        len(response.vehicles),
        response.license_plate,
        response.error,
    )

    return response
