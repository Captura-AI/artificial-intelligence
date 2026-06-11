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
    2. Detect plate region with platdetect.pt
    3. Widen the plate crop and read characters with platreader.pt
    4. Detect motorcycle type with motortype.pt and classify its color with
       color.pt (always runs, so a photo without a readable plate still gets
       searchable attributes)
    5. Save the annotated image to local disk
    6. Persist plate text + motor type/color in the database (a row is saved
       even when no plate is found, as long as a motorcycle is detected)

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

    # Step 2 — detect motorcycle type + color as alternative attributes
    motor_results: list[MotorAttributeResult] = []
    try:
        motor_results = run_motor_attributes(
            image=image,
            motortype_model_path=settings.motortype_model_path,
            color_model_path=settings.color_model_path,
            confidence_threshold=settings.motortype_confidence_threshold,
        )
    except Exception as exc:
        errors.append(f"Motor attribute pipeline failed: {exc}")

    readable_plates = [result for result in plate_results if result.text]
    dominant_motor = motor_results[0] if motor_results else None

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

    # Step 4 — persist results; a scan without plate text is still recorded
    # when a motorcycle was detected, so the upload stays searchable.
    try:
        for result in readable_plates:
            save_plate_result(
                database_url=settings.database_url,
                moment_id=uploader_id,
                file_path=annotated_image_path,
                plate_text=result.text,
                confidence=result.text_confidence,
                motor_type=dominant_motor.motor_type if dominant_motor else None,
                motor_type_confidence=dominant_motor.motor_type_confidence if dominant_motor else None,
                color=dominant_motor.color if dominant_motor else None,
                color_confidence=dominant_motor.color_confidence if dominant_motor else None,
            )
            plates.append(result.text)

        if not readable_plates and dominant_motor:
            save_plate_result(
                database_url=settings.database_url,
                moment_id=uploader_id,
                file_path=annotated_image_path,
                plate_text=None,
                confidence=None,
                motor_type=dominant_motor.motor_type,
                motor_type_confidence=dominant_motor.motor_type_confidence,
                color=dominant_motor.color,
                color_confidence=dominant_motor.color_confidence,
            )
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
                motor_type=result.motor_type,
                motor_type_confidence=result.motor_type_confidence,
                color=result.color,
                color_confidence=result.color_confidence,
                bbox=result.bbox,
            )
            for result in motor_results
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

