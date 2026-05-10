from dataclasses import dataclass
from typing import Optional

from PIL import Image

_MODEL = None
_MODEL_PATH: str = ""


@dataclass
class PlateCharacterDetection:
    text: str
    confidence: float
    bbox: list[int]


def _get_model(model_path: str):
    """Lazy-load platreader.pt YOLOv8 character-detection model on first use."""
    global _MODEL, _MODEL_PATH
    if _MODEL is None or model_path != _MODEL_PATH:
        from ultralytics import YOLO
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


def read_plate_characters(
    image: Image.Image,
    model_path: str = "platreader.pt",
    confidence_threshold: float = 0.3,
) -> list[PlateCharacterDetection]:
    """
    Detect characters from a cropped license plate image using platreader.pt.

    Returns local character boxes in the cropped plate coordinate system.
    """
    model = _get_model(model_path)
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    results = model(rgb_image, conf=confidence_threshold, verbose=False)

    chars: list[PlateCharacterDetection] = []

    for result in results:
        names = result.names  # {class_id: char_label}
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            char_label = names.get(cls_id, "?")
            chars.append(
                PlateCharacterDetection(
                    text=char_label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                )
            )

    chars.sort(key=lambda item: item.bbox[0])
    return chars


def read_plate_text(
    image: Image.Image,
    model_path: str = "platreader.pt",
    confidence_threshold: float = 0.3,
) -> tuple[Optional[str], Optional[float]]:
    """
    Read text from a cropped license plate image using platreader.pt.

    platreader.pt is a YOLOv8 character-detection model whose class names are
    the individual characters (0-9, A-Z). Detections are sorted left-to-right
    by bounding-box x-centre to reconstruct the plate string.

    Returns:
        (plate_text, mean_confidence) or (None, None) if no characters detected.
    """
    chars = read_plate_characters(
        image,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )

    if not chars:
        return None, None

    plate_text = "".join(char.text for char in chars).upper()
    mean_conf = round(sum(char.confidence for char in chars) / len(chars), 4)
    return plate_text, mean_conf


def is_model_ready(model_path: str = "platreader.pt") -> bool:
    """Check whether platreader.pt can be loaded without errors."""
    try:
        _get_model(model_path)
        return True
    except Exception:
        return False
