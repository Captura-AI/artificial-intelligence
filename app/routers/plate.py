import io
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image

from ..config import Settings, get_settings
from ..db.crud import delete_plate_results, save_plate_result
from ..models.schemas import (
    MotorDetection,
    PlateConfirmRequest,
    PlateConfirmResponse,
    PlateScanResponse,
)
from ..services.alpr_pipeline import PlatePipelineResult, run_plate_alpr
from ..services.motor_attribute_pipeline import MotorAttributeResult, run_motor_attributes

router = APIRouter(prefix="/plate", tags=["Plate"])


def _save_image(image: Image.Image, save_dir: str, uploader_id: str) -> str:
    """Save an image to disk and return its absolute path."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uploader_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir.resolve() / filename)
    image.convert("RGB").save(file_path, format="JPEG")
    return file_path


def _save_cv2_image(image_rgb: np.ndarray, save_dir: str, uploader_id: str) -> str:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uploader_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir.resolve() / filename)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image_bgr)
    return file_path


def _annotate_results(
    image: Image.Image,
    plate_results: list[PlatePipelineResult],
    motor_results: list[MotorAttributeResult],
) -> np.ndarray:
    annotated = np.array(image.convert("RGB"), dtype=np.uint8).copy()
    _draw_motor_results(annotated, motor_results)
    _draw_plate_results(annotated, plate_results)
    return annotated


def _draw_motor_results(annotated: np.ndarray, motor_results: list[MotorAttributeResult]) -> None:
    for motor_result in motor_results:
        x1, y1, x2, y2 = motor_result.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

        label = motor_result.motor_type
        if motor_result.color:
            label = f"{label} / {motor_result.color}"

        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_top = max(0, y1 - 35)
        label_right = x1 + max(200, text_width + 10)
        cv2.rectangle(annotated, (x1, label_top), (label_right, y1), (255, 255, 255), -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )


def _draw_plate_results(annotated: np.ndarray, plate_results: list[PlatePipelineResult]) -> None:
    for plate_result in plate_results:
        x1, y1, x2, y2 = plate_result.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if plate_result.text:
            (text_width, text_height), _ = cv2.getTextSize(
                plate_result.text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                2,
            )
            label_top = max(0, y1 - 35)
            label_right = x1 + max(200, text_width + 10)
            cv2.rectangle(annotated, (x1, label_top), (label_right, y1), (255, 255, 255), -1)
            cv2.putText(
                annotated,
                plate_result.text,
                (x1 + 5, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

        for character in plate_result.characters:
            cx1, cy1, cx2, cy2 = character.bbox
            gx1 = x1 + cx1
            gy1 = y1 + cy1
            gx2 = x1 + cx2
            gy2 = y1 + cy2
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 1)
            cv2.putText(
                annotated,
                character.text,
                (gx1, max(12, gy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )


def _match_plate_to_motor(
    plate_bbox: list[int],
    motor_results: list[MotorAttributeResult],
) -> Optional[int]:
    """
    Return the index of the motorcycle whose box contains the plate's center.

    When several boxes overlap the plate, the tightest (smallest-area) one wins,
    so a plate is paired with the closest-fitting motorcycle. Returns None when
    the plate falls outside every detected motorcycle (e.g. a car plate).
    """
    px = (plate_bbox[0] + plate_bbox[2]) / 2
    py = (plate_bbox[1] + plate_bbox[3]) / 2

    best_index: Optional[int] = None
    best_area: Optional[int] = None
    for index, motor in enumerate(motor_results):
        mx1, my1, mx2, my2 = motor.bbox
        if mx1 <= px <= mx2 and my1 <= py <= my2:
            area = (mx2 - mx1) * (my2 - my1)
            if best_area is None or area < best_area:
                best_index = index
                best_area = area

    return best_index


def _delete_previous_temp_images(settings: Settings, uploader_id: str) -> None:
    directory = Path(settings.plate_save_dir)
    if not directory.exists():
        return

    for image_path in directory.glob(f"{uploader_id}_*"):
        if image_path.is_file():
            image_path.unlink(missing_ok=True)


@router.post("/scan", response_model=PlateScanResponse, summary="Scan an uploaded image for a license plate")
async def scan_plate(
    uploader_id: str,
    file: UploadFile = File(..., description="Image file to scan"),
    settings: Settings = Depends(get_settings),
) -> PlateScanResponse:
    """
    Main entry point for plate recognition.

    Flow:
    1. Receive uploaded image
    2. Detect plate region with platdetect.pt, then widen the crop and read
       characters with platreader.pt
    3. Detect motorcycle type with motortype.pt and classify its color with
       color.pt (both always run, independent of the plate stage)
    4. Pair each plate with the motorcycle it sits on so one record holds the
       full set of attributes (plate + type + color); a motorcycle without a
       readable plate keeps its type/color, and a plate without a motorcycle is
       still recorded on its own
    5. Save the annotated image to local disk
    6. Persist one row per motorcycle (with its matched plate) plus one row per
       unmatched plate, so the upload stays searchable by either attribute

    The NestJS BE can then call GET /plate/search?text=<plate> to look it up.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        return PlateScanResponse(uploader_id=uploader_id, error=f"Cannot open image: {exc}")

    try:
        _delete_previous_temp_images(settings, uploader_id)
        delete_plate_results(settings.database_url, uploader_id)
    except Exception as exc:
        return PlateScanResponse(
            uploader_id=uploader_id,
            error=f"Failed to clear previous temp images: {exc}",
        )

    saved_photo = _save_image(image, settings.photo_save_dir, uploader_id)

    confidence: Optional[float] = None
    plates: list[str] = []
    saved_result_photo: Optional[str] = None
    errors: list[str] = []

    # Step 1 — detect plate bounding box(es) and read their text
    plate_results: list[PlatePipelineResult] = []
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
        errors.append(f"Plate pipeline failed: {exc}")

    # Step 2 — detect motorcycle type + color. Detect down to the lower assist
    # threshold so a bike that scores below the standalone bar can still be
    # recovered when a plate corroborates it (see filtering below).
    motor_candidates: list[MotorAttributeResult] = []
    try:
        motor_candidates = run_motor_attributes(
            image=image,
            motortype_model_path=settings.motortype_model_path,
            color_model_path=settings.color_model_path,
            confidence_threshold=settings.motortype_assist_confidence_threshold,
        )
    except Exception as exc:
        errors.append(f"Motor attribute pipeline failed: {exc}")

    readable_plates = [result for result in plate_results if result.text]

    # Pair each plate with the motorcycle candidate it sits on, so one record
    # ends up with the full set of attributes (plate + type + color). A bike
    # keeps the highest-confidence plate matched to it; plates that match no
    # candidate are tracked separately and saved on their own (e.g. car plates).
    plate_for_candidate: dict[int, PlatePipelineResult] = {}
    matched_plate_ids: set[int] = set()
    for plate_index, plate in enumerate(readable_plates):
        candidate_index = _match_plate_to_motor(plate.bbox, motor_candidates)
        if candidate_index is None:
            continue
        matched_plate_ids.add(plate_index)
        current = plate_for_candidate.get(candidate_index)
        if current is None or (plate.text_confidence or 0.0) > (current.text_confidence or 0.0):
            plate_for_candidate[candidate_index] = plate

    # Keep a candidate as a real motorcycle when it clears the standalone
    # threshold on its own, or when a plate inside it confirms a vehicle.
    motor_results: list[MotorAttributeResult] = []
    plate_for_motor: dict[int, PlatePipelineResult] = {}
    for candidate_index, candidate in enumerate(motor_candidates):
        clears_threshold = candidate.motor_type_confidence >= settings.motortype_confidence_threshold
        matched_plate = plate_for_candidate.get(candidate_index)
        if not clears_threshold and matched_plate is None:
            continue
        if matched_plate is not None:
            plate_for_motor[len(motor_results)] = matched_plate
        motor_results.append(candidate)

    unmatched_plates = [
        plate for plate_index, plate in enumerate(readable_plates) if plate_index not in matched_plate_ids
    ]

    # Step 3 — annotate whatever was found (plates and/or motorcycles)
    annotated_image_path: Optional[str] = None
    if readable_plates or motor_results:
        try:
            annotated_image = _annotate_results(image, readable_plates, motor_results)
            annotated_image_path = _save_cv2_image(
                annotated_image,
                settings.plate_save_dir,
                f"{uploader_id}_annotated",
            )
            saved_result_photo = annotated_image_path
        except Exception as exc:
            errors.append(f"Failed to save annotated image: {exc}")

    # Step 4 — persist one row per motorcycle (with its matched plate, if any),
    # plus one row per plate that did not belong to any detected motorcycle.
    try:
        for motor_index, motor in enumerate(motor_results):
            matched_plate = plate_for_motor.get(motor_index)
            save_plate_result(
                database_url=settings.database_url,
                moment_id=uploader_id,
                file_path=annotated_image_path,
                plate_text=matched_plate.text if matched_plate else None,
                confidence=matched_plate.text_confidence if matched_plate else None,
                motor_type=motor.motor_type,
                motor_type_confidence=motor.motor_type_confidence,
                color=motor.color,
                color_confidence=motor.color_confidence,
            )
            if matched_plate and matched_plate.text:
                plates.append(matched_plate.text)

        for plate in unmatched_plates:
            save_plate_result(
                database_url=settings.database_url,
                moment_id=uploader_id,
                file_path=annotated_image_path,
                plate_text=plate.text,
                confidence=plate.text_confidence,
            )
            if plate.text:
                plates.append(plate.text)
    except Exception as exc:
        errors.append(f"Failed to save results to database: {exc}")

    if plates:
        confidence = max((result.text_confidence or 0.0) for result in readable_plates)
    elif motor_results:
        errors.append("No plate text detected.")
    else:
        errors.append("No plate or motorcycle detected.")

    return PlateScanResponse(
        uploader_id=uploader_id,
        confidence=confidence,
        plates=plates,
        motors=[
            MotorDetection(
                motor_type=motor.motor_type,
                motor_type_confidence=motor.motor_type_confidence,
                color=motor.color,
                color_confidence=motor.color_confidence,
                plate=plate_for_motor[motor_index].text if motor_index in plate_for_motor else None,
                plate_confidence=(
                    plate_for_motor[motor_index].text_confidence if motor_index in plate_for_motor else None
                ),
                bbox=motor.bbox,
            )
            for motor_index, motor in enumerate(motor_results)
        ],
        saved_photo=Path(saved_photo).name if saved_photo else None,
        saved_result_photo=Path(saved_result_photo).name if saved_result_photo else None,
        error="; ".join(errors) if errors else None,
    )


@router.get("/result/{filename}", summary="Serve a saved annotated plate image by filename")
async def get_result_image(
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    file_path = Path(settings.plate_save_dir).resolve() / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(str(file_path), media_type="image/jpeg")


@router.post("/confirm", response_model=PlateConfirmResponse, summary="Save or discard an uploaded plate scan")
async def confirm_plate(
    body: PlateConfirmRequest,
    settings: Settings = Depends(get_settings),
) -> PlateConfirmResponse:
    """
    Confirm whether to keep or discard files from a previous /plate/scan call.

    - action "save"    → files are kept as-is, no changes made.
    - action "discard" → deletes the specific files by filename and removes
                         plate_results DB records for the uploader_id.

    The filenames must be taken directly from the saved_photo / saved_result_photo
    fields returned by /plate/scan (just the basename, not the full path).
    """
    if body.action not in ("save", "discard"):
        return PlateConfirmResponse(
            uploader_id=body.uploader_id,
            action=body.action,
            success=False,
            message="Invalid action. Must be 'save' or 'discard'.",
        )

    if body.action == "save":
        return PlateConfirmResponse(
            uploader_id=body.uploader_id,
            action=body.action,
            success=True,
            message="Photo saved successfully.",
        )

    # action == "discard"
    errors: list[str] = []

    if body.saved_photo_filename:
        photo_path = Path(settings.photo_save_dir).resolve() / body.saved_photo_filename
        if photo_path.exists() and photo_path.is_file():
            photo_path.unlink(missing_ok=True)
        elif not photo_path.exists():
            errors.append(f"Photo file not found: {body.saved_photo_filename}")

    if body.saved_result_photo_filename:
        result_path = Path(settings.plate_save_dir).resolve() / body.saved_result_photo_filename
        if result_path.exists() and result_path.is_file():
            result_path.unlink(missing_ok=True)
        elif not result_path.exists():
            errors.append(f"Result photo file not found: {body.saved_result_photo_filename}")

    try:
        delete_plate_results(settings.database_url, body.uploader_id)
    except Exception as exc:
        errors.append(f"Failed to delete DB records: {exc}")

    if errors:
        return PlateConfirmResponse(
            uploader_id=body.uploader_id,
            action=body.action,
            success=False,
            message="; ".join(errors),
        )

    return PlateConfirmResponse(
        uploader_id=body.uploader_id,
        action=body.action,
        success=True,
        message="Photo discarded successfully.",
    )

