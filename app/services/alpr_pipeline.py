from typing import Optional

from PIL import Image

from ..models.internal import PlatePipelineResult
from .geometry import _expand_bbox
from .plate_detector import detect_plates
from .plate_text_reader import read_plate_characters

# Re-exported so existing callers can keep importing the DTO from this module.
__all__ = ["PlatePipelineResult", "run_plate_alpr", "find_best_plate"]


def run_plate_alpr(
    image: Image.Image,
    detector_model_path: str,
    reader_model_path: str,
    detector_confidence_threshold: float,
    reader_confidence_threshold: float,
    padding_px: int,
) -> list[PlatePipelineResult]:
    """
    Run the two-stage ALPR flow used in Colab.

    1. Detect plate boxes with platdetect.pt.
    2. Widen each detected plate crop by a fixed padding.
    3. Read per-character detections from platreader.pt.
    4. Reconstruct raw plate text left-to-right.
    """
    plate_boxes = detect_plates(
        image,
        model_path=detector_model_path,
        confidence_threshold=detector_confidence_threshold,
    )

    results: list[PlatePipelineResult] = []
    for detection_confidence, bbox in plate_boxes:
        padded_bbox = _expand_bbox(bbox, image.size, padding_px)
        plate_crop = image.crop(tuple(padded_bbox))
        characters = read_plate_characters(
            plate_crop,
            model_path=reader_model_path,
            confidence_threshold=reader_confidence_threshold,
        )
        plate_text = None
        text_confidence = None
        if characters:
            plate_text = "".join(character.text for character in characters).upper()
            text_confidence = round(
                sum(character.confidence for character in characters) / len(characters),
                4,
            )
        results.append(
            PlatePipelineResult(
                text=plate_text,
                text_confidence=text_confidence,
                detection_confidence=detection_confidence,
                bbox=padded_bbox,
                crop=plate_crop,
                characters=characters,
            )
        )

    return results


def find_best_plate(
    image: Image.Image,
    detector_model_path: str,
    reader_model_path: str,
    detector_confidence_threshold: float,
    reader_confidence_threshold: float,
    padding_px: int,
) -> Optional[PlatePipelineResult]:
    """
    Return the first padded plate detection that produces character output.

    Detections are already sorted by detector confidence, so this preserves the
    current API contract of returning a single best plate while using the wider
    crop + character ordering logic from the Colab pipeline.
    """
    candidates = run_plate_alpr(
        image=image,
        detector_model_path=detector_model_path,
        reader_model_path=reader_model_path,
        detector_confidence_threshold=detector_confidence_threshold,
        reader_confidence_threshold=reader_confidence_threshold,
        padding_px=padding_px,
    )

    for candidate in candidates:
        if candidate.text:
            return candidate

    return None
