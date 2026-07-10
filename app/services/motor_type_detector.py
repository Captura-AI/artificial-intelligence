from PIL import Image

from .model_cache import is_model_ready as _cache_is_model_ready
from .model_cache import load_cached_model
from .utils import ensure_rgb


def _load_model(model_path: str):
    """Load the motortype.pt YOLOv8 model from ``model_path``."""
    from ultralytics import YOLO

    return YOLO(model_path)


def _get_model(model_path: str):
    """Lazy-load motortype.pt YOLOv8 model on first use (cached per path)."""
    return load_cached_model(model_path, _load_model)


def detect_motor_types(
    image: Image.Image,
    model_path: str = "motortype.pt",
    *,
    confidence_threshold: float,
) -> list[tuple[str, float, list[int]]]:
    """
    Detect motorcycles and classify their body style using motortype.pt.

    Returns:
        List of (motor_type, confidence, bbox) tuples sorted by confidence
        descending, where bbox is [x1, y1, x2, y2] in pixel coordinates and
        motor_type is one of the model classes (Cruiser, Dual-Sport,
        Motocross, Scooter, Sport, Standard, Touring).
    """
    model = _get_model(model_path)
    rgb_image = ensure_rgb(image)
    results = model(rgb_image, conf=confidence_threshold, verbose=False)

    detections: list[tuple[str, float, list[int]]] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0].item())
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0].item())
            bbox = [int(v) for v in box.xyxy[0].tolist()]
            motor_type = model.names.get(cls_id, "Unknown")
            detections.append((motor_type, round(conf, 4), bbox))

    detections.sort(key=lambda x: x[1], reverse=True)
    return detections


def is_model_ready(model_path: str = "motortype.pt") -> bool:
    """Check whether motortype.pt can be loaded without errors."""
    return _cache_is_model_ready(model_path, _load_model)
