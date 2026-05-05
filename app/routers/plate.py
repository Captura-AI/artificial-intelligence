import io
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File
from PIL import Image

from ..config import Settings, get_settings
from ..db.crud import save_plate_result
from ..models.schemas import PlateScanResponse
from ..services.plate_detector import detect_plates
from ..services.plate_text_reader import read_plate_text

router = APIRouter(prefix="/plate", tags=["Plate"])


def _save_image(image: Image.Image, save_dir: str, moment_id: str) -> str:
    """Save the plate crop to disk and return its absolute path."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{moment_id}_{uuid.uuid4().hex}.jpg"
    file_path = str(out_dir.resolve() / filename)
    image.convert("RGB").save(file_path, format="JPEG")
    return file_path


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
    3. Read plate text with platreader.pt
    4. Save the plate crop image to local disk
    5. Persist file_path + plate_text in the database

    The NestJS BE can then call GET /plate/search?text=<plate> to look it up.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        return PlateScanResponse(moment_id=moment_id, error=f"Cannot open image: {exc}")

    # Step 1 — detect plate bounding box(es) in the full image
    plate_text: Optional[str] = None
    confidence: Optional[float] = None
    file_path: Optional[str] = None
    error: Optional[str] = None

    try:
        plate_boxes = detect_plates(
            image,
            model_path=settings.platdetect_model_path,
            confidence_threshold=settings.platdetect_confidence_threshold,
        )

        # Use the highest-confidence detected region; fall back to full image
        if plate_boxes:
            _, p_bbox = plate_boxes[0]
            plate_crop = image.crop((p_bbox[0], p_bbox[1], p_bbox[2], p_bbox[3]))
        else:
            plate_crop = image

        # Step 2 — read text from the plate crop
        plate_text, confidence = read_plate_text(
            plate_crop,
            model_path=settings.platreader_model_path,
            confidence_threshold=settings.plate_confidence_threshold,
        )

        if plate_text:
            # Step 3 — save crop to disk
            file_path = _save_image(plate_crop, settings.plate_save_dir, moment_id)

            # Step 4 — write to database
            save_plate_result(
                database_url=settings.database_url,
                moment_id=moment_id,
                file_path=file_path,
                plate_text=plate_text,
                confidence=confidence,
            )
        else:
            error = "No plate text detected."

    except Exception as exc:
        error = str(exc)

    return PlateScanResponse(
        moment_id=moment_id,
        plate_text=plate_text,
        confidence=confidence,
        file_path=file_path,
        error=error,
    )



