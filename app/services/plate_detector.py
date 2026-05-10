from typing import Optional

from PIL import Image

_MODEL = None
_MODEL_PATH: str = ""


def _get_model(model_path: str):
    """Lazy-load platdetect.pt YOLOv8 model on first use."""
    global _MODEL, _MODEL_PATH
    if _MODEL is None or model_path != _MODEL_PATH:
        from ultralytics import YOLO
        _MODEL = YOLO(model_path)
        _MODEL_PATH = model_path
    return _MODEL


def detect_plates(
    image: Image.Image,
    model_path: str = "platdetect.pt",
    confidence_threshold: float = 0.25,
) -> list[tuple[float, list[int]]]:
    """
    Detect license plate bounding boxes in an image using platdetect.pt.

    Returns:
        List of (confidence, bbox) tuples sorted by confidence descending,
        where bbox is [x1, y1, x2, y2] in pixel coordinates.
    """
    model = _get_model(model_path)
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    results = model(rgb_image, conf=confidence_threshold, verbose=False)

    plates: list[tuple[float, list[int]]] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf < confidence_threshold:
                continue
            bbox = [int(v) for v in box.xyxy[0].tolist()]
            plates.append((round(conf, 4), bbox))

    plates.sort(key=lambda x: x[0], reverse=True)
    return plates


def is_model_ready(model_path: str = "platdetect.pt") -> bool:
    """Check whether platdetect.pt can be loaded without errors."""
    try:
        _get_model(model_path)
        return True
    except Exception:
        return False
