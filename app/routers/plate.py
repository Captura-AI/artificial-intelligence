import io
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File
from PIL import Image

from ..config import Settings, get_settings
from ..db.crud import save_plate_result
from ..models.schemas import PlateScanResponse
from ..services.alpr_pipeline import PlatePipelineResult, run_plate_alpr

router = APIRouter(prefix="/plate", tags=["Plate"])


def _save_image(image: Image.Image, save_dir: str, moment_id: str) -> str:
    """Save an image to disk and return its absolute path."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{moment_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir.resolve() / filename)
    image.convert("RGB").save(file_path, format="JPEG")
    return file_path


def _save_cv2_image(image_rgb: np.ndarray, save_dir: str, moment_id: str) -> str:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{moment_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir.resolve() / filename)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image_bgr)
    return file_path


def _annotate_plate_results(image: Image.Image, plate_results: list[PlatePipelineResult]) -> np.ndarray:
    annotated = np.array(image.convert("RGB"), dtype=np.uint8).copy()

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

    return annotated


@router.post("/scan", response_model=PlateScanResponse, summary="Scan an uploaded image for a license plate")
async def scan_plate(
    moment_id: str,
    file: UploadFile = File(..., description="Image file to scan"),
    settings: Settings = Depends(get_settings),
) -> PlateScanResponse:
    """
    Main entry point for plate recognition.

    Flow:
    1. Receive uploaded image
    2. Detect plate region with platdetect.pt
    3. Widen the plate crop and read characters with platreader.pt
    4. Save the plate crop image to local disk
    5. Persist file_path + plate_text in the database

    The NestJS BE can then call GET /plate/search?text=<plate> to look it up.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        return PlateScanResponse(uploader_id=moment_id, error=f"Cannot open image: {exc}")

    saved_photo = _save_image(image, settings.photo_save_dir, moment_id)

    # Step 1 — detect plate bounding box(es) in the full image
    confidence: Optional[float] = None
    plates: list[str] = []
    saved_result_photo: Optional[str] = None
    error: Optional[str] = None

    try:
        results = run_plate_alpr(
            image=image,
            detector_model_path=settings.platdetect_model_path,
            reader_model_path=settings.platreader_model_path,
            detector_confidence_threshold=settings.platdetect_confidence_threshold,
            reader_confidence_threshold=settings.plate_confidence_threshold,
            padding_px=settings.plate_padding_px,
        )

        annotated_results = [result for result in results if result.text]
        annotated_image_path: Optional[str] = None
        if annotated_results:
            annotated_image = _annotate_plate_results(image, annotated_results)
            annotated_image_path = _save_cv2_image(
                annotated_image,
                settings.plate_save_dir,
                f"{moment_id}_annotated",
            )

        for result in results:
            if not result.text:
                continue

            save_plate_result(
                database_url=settings.database_url,
                moment_id=moment_id,
                file_path=annotated_image_path,
                plate_text=result.text,
                confidence=result.text_confidence,
            )
            plates.append(result.text)

        if plates:
            confidence = max(
                (result.text_confidence or 0.0) for result in results if result.text
            )
            saved_result_photo = annotated_image_path
        else:
            error = "No plate text detected."

    except Exception as exc:
        error = str(exc)

    return PlateScanResponse(
        uploader_id=moment_id,
        confidence=confidence,
        plates=plates,
        saved_photo=saved_photo,
        saved_result_photo=saved_result_photo,
        error=error,
    )



