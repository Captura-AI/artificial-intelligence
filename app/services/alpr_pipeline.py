from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .plate_detector import detect_plates
from .plate_text_reader import PlateCharacterDetection, read_plate_characters


@dataclass
class PlatePipelineResult:
    text: Optional[str]
    text_confidence: Optional[float]
    detection_confidence: float
    bbox: list[int]
    crop: Image.Image
    characters: list[PlateCharacterDetection]


def _expand_bbox(bbox: list[int], image_size: tuple[int, int], padding_px: int) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [
        max(0, int(x1) - padding_px),
        max(0, int(y1) - padding_px),
        min(width, int(x2) + padding_px),
        min(height, int(y2) + padding_px),
    ]


def run_plate_alpr(
    image: Image.Image,
    detector_model_path: str,
    reader_model_path: str,
    detector_confidence_threshold: float,
    reader_confidence_threshold: float,
    padding_px: int = 15,
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
    padding_px: int = 15,
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