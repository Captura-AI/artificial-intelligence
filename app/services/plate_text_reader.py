from typing import Optional

from PIL import Image

_MODEL = None
_MODEL_PATH: str = ""


def _get_model(model_path: str):
    """Lazy-load platreader.pt YOLOv8 character-detection model on first use."""
    global _MODEL, _MODEL_PATH
    if _MODEL is None or model_path != _MODEL_PATH:
        from ultralytics import YOLO
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


def read_plate_text(
    image: Image.Image,
    model_path: str = "platreader.pt",
    confidence_threshold: float = 0.4,
) -> tuple[Optional[str], Optional[float]]:
    """
    Read text from a cropped license plate image using platreader.pt.

    platreader.pt is a YOLOv8 character-detection model whose class names are
    the individual characters (0-9, A-Z). Detections are sorted left-to-right
    by bounding-box x-centre to reconstruct the plate string.

    Returns:
        (plate_text, mean_confidence) or (None, None) if no characters detected.
    """
    model = _get_model(model_path)
    results = model(image, verbose=False)

    # Each entry: (x_centre, confidence, char_label)
    chars: list[tuple[float, float, str]] = []

    for result in results:
        names = result.names  # {class_id: char_label}
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0].item())
            x1, _, x2, _ = box.xyxy[0].tolist()
            x_centre = (x1 + x2) / 2.0
            char_label = names.get(cls_id, "?")
            chars.append((x_centre, conf, char_label))

    if not chars:
        return None, None

    chars.sort(key=lambda c: c[0])
    plate_text = "".join(c[2] for c in chars).upper()
    mean_conf = round(sum(c[1] for c in chars) / len(chars), 4)
    return plate_text, mean_conf


def is_model_ready(model_path: str = "platreader.pt") -> bool:
    """Check whether platreader.pt can be loaded without errors."""
    try:
        _get_model(model_path)
        return True
    except Exception:
        return False
